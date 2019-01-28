import pandas as pd
import numpy as np
import cv2

# NOTE: If training only on CPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Dropout, Dense, BatchNormalization, Flatten, Reshape, Input, MaxPooling2D, merge, Activation, AveragePooling2D
from keras.preprocessing.image import load_img, flip_axis, img_to_array #"""TODO:""" random_shift, random_shear

import matplotlib.pyplot as plt

def _generator(csv_data, batch_size, input_shape):
    while 1:
        batch = csv_data.sample(n=batch_size)
        batch_X, batch_y = process_batch(batch, input_shape=input_shape)
        yield np.array(batch_X), np.array(batch_y)

def process_batch(batch, steering_offset=0.25, input_shape=(60, 60, 1)):
    batch_X, batch_y = [], []
    for row in batch.itertuples():
        c_img, l_img, r_img = load_img(row[1].strip(), target_size=input_shape), load_img(row[2].strip(), target_size=input_shape), load_img(row[3].strip(), target_size=input_shape)
        steering_angle = row[4]
        images, steering_angles = preprocess_image([c_img,l_img,r_img], steering_angle=steering_angle)
        batch_X += [images[0], images[1], images[2], flip_axis(images[0], 1), flip_axis(images[1], 1), flip_axis(images[2],1)]
        batch_y += [steering_angles[0], steering_angles[1]+steering_offset, steering_angles[2]-steering_offset, -steering_angles[0],  -(steering_angles[1]+steering_offset), -(steering_angles[2]-steering_offset)]
    return batch_X, batch_y

def preprocess_image(images, steering_angle=None):
    img_return, steering_angles = [], []
    for img in images:
        img = img_to_array(img)
        img = np.subtract(np.divide(img, 255), -0.5)
        img = random_darken(img)
        steering_angles.append(steering_angle)
        img_return.append(img)
    return img_return, steering_angles

def random_darken(image, frac_brightness=0.5):
    im_shape = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    start_x, start_y = np.random.randint(0, im_shape[0]), np.random.randint(0, im_shape[1])
    end_x, end_y = np.random.randint(start_x, im_shape[0]), np.random.randint(start_y, im_shape[1])
    hsv[start_x:end_x, start_y:end_y, 2] *= frac_brightness
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

def create_model(input_shape):
    convolutions = [32, 64, 128]
    fc_layers = [1000, 500, 250]
    # NOTE:
    # fc_layers = [750, 500, 250]
    # does not give good results
    # Incremented from [100,75,50]

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for conv in convolutions:
        model.add(Convolution2D(conv, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for fc in fc_layers:
        model.add(Dense(fc, activation='relu'))
        model.add(Dropout(0.6))
    model.add(Dense(1, activation='linear'))

    return model


def train():

    # NOTE: Max input_shape and batch size theat fits on local GPU
    input_shape = (60, 60, 3) # Sticking to around powers of 2, 64 failed, rounding
    batch_size = 64 # Sticking to powers of 2

    model = create_model(input_shape)
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    csv_data = pd.read_csv('./data/driving_log.csv')
    model.fit_generator(_generator(csv_data, batch_size=batch_size, input_shape=input_shape), samples_per_epoch=250*batch_size, nb_epoch=10)
    # Use generator, highly effective
    # Reference: https://github.com/CYHSM/carnd/blob/master/CarND-Behavioral-Cloning/behavioral_cloning.py
    # TODO: DEBUG: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
    # Occured only once

    model.save('model.h5')

if __name__ == '__main__':
    train()
