direct = "../input/stars-and-galaxies/data"
import os

import numpy as np

import pandas as pd

import pickle as pkl

import matplotlib.pyplot as plt

import matplotlib.image as img

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, ZeroPadding2D, LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
train_datagen = ImageDataGenerator(

        rescale=1./255,

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(

        f'{direct}/train',

        target_size=(32,32),

        batch_size=32,

        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(

        f'{direct}/validation',

        target_size=(32,32),

        batch_size=32,

        class_mode='categorical')
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(32,32,3)))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid'))

model.add(LeakyReLU(alpha=0.1))

model.add(ZeroPadding2D(padding=(1,1)))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(2048))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(0.5))



model.add(Dense(2048))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax'))



model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

es = EarlyStopping(monitor='val_loss', verbose=1, patience=10, restore_best_weights=True)



cb = [es]
model.fit(train_generator,

        epochs=50,

        callbacks = cb,

        validation_data=validation_generator)