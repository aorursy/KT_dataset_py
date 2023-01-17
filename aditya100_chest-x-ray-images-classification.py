import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob

import cv2

import tensorflow as tf

import keras

import keras.backend as k

import os



from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
train_dir = '../input/chest_xray/chest_xray/train/'

test_dir = '../input/chest_xray/chest_xray/test'

val_dir = '../input/chest_xray/chest_xray/val'

input_shape = (150, 150, 3)



from keras.preprocessing.image import ImageDataGenerator



gen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



train = gen.flow_from_directory(train_dir, (150, 150), shuffle=True, seed=1, batch_size=16)

val = gen.flow_from_directory(val_dir, (150, 150), shuffle=True, seed=1, batch_size=16)
from keras import layers

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('sigmoid'))
opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train, validation_data=val, epochs=1, steps_per_epoch=5217, validation_steps=17, verbose=1)
test = gen.flow_from_directory(test_dir, (150, 150), shuffle=False, seed=1, batch_size=8)



scores = model.evaluate_generator(test, steps = 624)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))