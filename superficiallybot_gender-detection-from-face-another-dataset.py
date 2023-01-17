# do the necessary imports



import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import random

import cv2

import os

from PIL import Image



from keras.layers import *

from keras.models import *

import keras

# Hyper - parameters



epochs = 100

lr = 1e-3

batch_size = 64

#img_dims = (96,96,3)



data = []

labels = []
size = 224
train_path = '../input/gender-classification-dataset/Training'

valid_path = '../input/gender-classification-dataset/Validation'
m = len(os.listdir(os.path.join(train_path, 'male')))

f = len(os.listdir(os.path.join(train_path, 'female')))

print(m+f, 'for training')
m = len(os.listdir(os.path.join(valid_path, 'male')))

f = len(os.listdir(os.path.join(valid_path, 'female')))

print(m+f, 'for validation')
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

    rescale = 1./255, 

    shear_range = 0.2, 

    zoom_range = 0.2,

    horizontal_flip = True

)



test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

    train_path, 

    target_size = (size, size),

    batch_size = batch_size,

    class_mode = 'binary'

)



validation_generator = test_datagen.flow_from_directory(

    valid_path,

    target_size = (size, size),

    batch_size = batch_size, 

    class_mode = 'binary'

)



def model1_smallervgg(size = 224):

    model = Sequential()

    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (size, size, 3) , activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((3,3)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(1024, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(1, activation = 'sigmoid'))

    opt = keras.optimizers.Adam(lr=lr, decay=lr/epochs)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    

    return model

    
model1 = model1_smallervgg()
model1.summary()
from keras.callbacks.callbacks import EarlyStopping

from keras.callbacks.callbacks import ReduceLROnPlateau
es = EarlyStopping(patience=5, monitor = 'val_accuracy')

rlp = ReduceLROnPlateau(patience=5, monitor = 'val_accuracy')



callbacks = [es, rlp]
steps_per_epoch = int(47009/128)

steps_per_epoch
model1.fit_generator(train_generator, steps_per_epoch = steps_per_epoch , epochs = 30, callbacks = callbacks, validation_data = validation_generator)
model1.save('model1.h5')