import glob

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2



train = np.empty(3, dtype=object)



for t in [1, 2, 3]:

    image_paths = glob.glob('../input/train/Type_{}/*.jpg'.format(t))

    train[t-1] = np.zeros((len(image_paths), 64, 64, 1))

    for i, img_path in enumerate(image_paths):

        # Take only the red channel

        img = cv2.resize(cv2.imread(img_path), (64, 64))[:, :, 2:]

        train[t-1][i, :] = img
# This function will pick random images from each of 

# the 3 classes, with random transformations

def image_generator(batch_size=60):

    n = batch_size // 3

    X = np.zeros((batch_size, 64, 64, 1))

    y = np.zeros((batch_size, 3))

    while True:

        for t in [0, 1, 2]:

            idx = np.random.choice(np.arange(train[t].shape[0]), n, replace=False)

            X[t * n : (t + 1) * n] = train[t][idx, :]

            y[t * n : (t + 1) * n, t] = 1



        yield X, y

g = image_generator()

x, y = next(g)

x.shape, y.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D



model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(256,256,1)))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.15))



model.add(Conv2D(64, (3,3)))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(128))

model.add(Dropout(0.5))



model.add(Dense(3))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



model.fit_generator(image_generator(), steps_per_epoch=30, epochs=10)