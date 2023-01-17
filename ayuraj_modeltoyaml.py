import numpy as np 

import pandas as pd

import os

import cv2

import random

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import shuffle



import keras.backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.utils import Sequence
def new_model(input_shape):

    inputs = Input(shape=input_shape)

    

    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(inputs)

    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = Dropout(0.25)(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

    

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(x)

    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')(x)

    x = MaxPooling2D(pool_size=(2,2))(x)

        

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dense(256, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    age = Dense(5, activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=age)

    return model
model = new_model((73, 81, 3))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_yaml = model.to_yaml()

with open("model_age.yaml", "w") as yaml_file:

    yaml_file.write(model_yaml)

# serialize weights to HDF5

# model.save_weights("model.h5")

print("Saved model to disk")