import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.image import imread

import pathlib

import cv2

import os



from PIL import Image

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential

from keras.optimizers import SGD

from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#constants

IMG_HEIGTH = 256

IMG_WEIGTH = 256

def model_init():

    cnn = Sequential()

    cnn.add(Conv2D(filters=256, 

                   kernel_size=(3,3), 

                   strides=(1,1),

                   padding='same',

                   input_shape=(IMG_HEIGTH,IMG_WEIGTH,3),

                   data_format='channels_last'))

    cnn.add(BatchNormalization())

    cnn.add(Activation('relu'))

    cnn.add(MaxPooling2D(pool_size=(2,2),

                         strides=2))

    cnn.add(Conv2D(filters=128,

                   kernel_size=(3,3),

                   strides=(1,1),

                   padding='same'))

    cnn.add(BatchNormalization())

    cnn.add(Activation('relu'))

    cnn.add(MaxPooling2D(pool_size=(2,2),

                         strides=2))

    cnn.add(Conv2D(filters=64,

                   kernel_size=(3,3),

                   strides=(1,1),

                   padding='same'))

    cnn.add(BatchNormalization())

    cnn.add(Activation('relu'))

    cnn.add(MaxPooling2D(pool_size=(2,2),

                         strides=2))

    cnn.add(Conv2D(filters=32,

                   kernel_size=(3,3),

                   strides=(1,1),

                   padding='same'))

    cnn.add(BatchNormalization())

    cnn.add(Activation('relu'))

    cnn.add(MaxPooling2D(pool_size=(2,2),

                         strides=2))

    

    cnn.add(Flatten())        

    cnn.add(Dense(128))

    cnn.add(Activation('relu'))

    cnn.add(Dropout(0.5))

    cnn.add(Dense(10))

    cnn.add(Activation('softmax'))

    

    cnn.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    

    return cnn
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        fill_mode='reflect',

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        '../input/training/training',

        target_size=(IMG_HEIGTH, IMG_WEIGTH),

        batch_size=32,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        '../input/validation/validation',

        target_size=(IMG_HEIGTH, IMG_WEIGTH),

        batch_size=32,

        class_mode='binary')
model = model_init()

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(

            train_generator,

            steps_per_epoch=1097/8,

            epochs=30,

            validation_data=validation_generator,

            validation_steps=272/32)
img = Image.open('../input/validation/validation/n1/n100.jpg').resize((IMG_HEIGTH,IMG_WEIGTH))

np_img = np.expand_dims(np.array(img), axis=0)

print(np_img.shape)

plt.imshow(np_img[0])

result = model.predict(np_img)

print (result)
!ls ../input/training/training/n0/
model.save("model_30epoch.h5")