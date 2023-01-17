import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



import os



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))




EPOCHS = 20

INIT_LR = 1e-5

BS = 8

default_image_size = tuple((256, 256))

image_size = 0



width=256

height=256

depth=3
#since number of classes being used is 3

n_classes = 3
from keras import layers

from keras.models import Model



optss = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

def alexnet(in_shape=(256,256,3), n_classes=n_classes, opt=optss):

    in_layer = layers.Input(in_shape)

    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)

    pool1 = layers.MaxPool2D(3, 2)(conv1)

    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)

    pool2 = layers.MaxPool2D(3, 2)(conv2)

    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)

    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)

    pool3 = layers.MaxPool2D(3, 2)(conv4)

    flattened = layers.Flatten()(pool3)

    dense1 = layers.Dense(4096, activation='relu')(flattened)

    drop1 = layers.Dropout(0.8)(dense1)

    dense2 = layers.Dense(4096, activation='relu')(drop1)

    drop2 = layers.Dropout(0.8)(dense2)

    preds = layers.Dense(n_classes, activation='softmax')(drop2)



    model = Model(in_layer, preds)

    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    return model



model = alexnet()
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png')
model.summary()