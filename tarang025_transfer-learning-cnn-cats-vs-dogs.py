import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import os

import cv2

import csv

import shutil

from glob import glob

from PIL import Image

from IPython.display import FileLink
print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/"))
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip
!ls train/ | wc -l
os.mkdir('valid/')
os.mkdir('train/cats/')

os.mkdir('train/dogs/')

os.mkdir('valid/cats/')

os.mkdir('valid/dogs/')
cats = glob('train/cat*.jpg')

shuf = np.random.permutation(cats)



for i in range(len(cats)): shutil.move(shuf[i], 'train/cats/')
!ls train/cats | wc -l
!ls train/dogs | wc -l
!ls train/ | wc -l
dogs = glob('train/dog*.jpg')

shuf = np.random.permutation(dogs)



for i in range(len(dogs)): shutil.move(shuf[i], 'train/dogs/')
!ls train/dogs | wc -l
!ls train/ | wc -l
cats_valid = glob('train/cats/*.jpg')

shuf = np.random.permutation(cats_valid)



for i in range(int(len(cats_valid) / 10)): shutil.move(shuf[i], 'valid/cats/')
dogs_valid = glob('train/dogs/*.jpg')

shuf = np.random.permutation(dogs_valid)



for i in range(int(len(dogs_valid) / 10)): shutil.move(shuf[i], 'valid/dogs/')
!ls train/cats | wc -l
!ls train/dogs | wc -l
from __future__ import division, print_function



import os, json

from glob import glob

import numpy as np

from scipy import misc, ndimage

from scipy.ndimage.interpolation import zoom



from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.utils.data_utils import get_file

from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D

from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD, RMSprop, Adam

from keras.preprocessing import image

from keras.models import model_from_json

!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

path = 'data/'
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

vgg_mean.shape
def vgg_preprocess(x):

    """

        Subtracts the mean RGB value, and transposes RGB to BGR.

        The mean RGB was computed on the image set used to train the VGG model.

        Args: 

            x: Image array (height x width x channels)

        Returns:

            Image array (height x width x transposed_channels)

    """

    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

    x = x - vgg_mean

    return x[:, ::-1] # reverse axis rgb->bgr
model = Sequential()

model.add(Lambda(vgg_preprocess, input_shape = (224, 224, 3), output_shape = (224, 224, 3)))



model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))

model.add(MaxPooling2D((2, 2), strides = (2, 2)))



model.add(Flatten())

model.add(Dense(4096, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(4096, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1000, activation = 'softmax'))



model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])



fname = 'weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

model.load_weights(fname)
model.summary()
model.pop()

for layer in model.layers[:-1]: layer.trainable = False

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
batch_size = 128
datagen = image.ImageDataGenerator()

trn_batches = datagen.flow_from_directory('train/', target_size = (224, 224),

            class_mode = 'categorical', shuffle = True, batch_size = batch_size)



val_batches = datagen.flow_from_directory('valid/', target_size = (224, 224),

            class_mode = 'categorical', shuffle = True, batch_size = batch_size)

model.fit_generator(trn_batches, steps_per_epoch = trn_batches.n / batch_size, epochs = 1, validation_data = val_batches, 

                    validation_steps = val_batches.n / batch_size)
model.fit_generator(trn_batches, steps_per_epoch = trn_batches.n / batch_size, epochs = 5, validation_data = val_batches, 

                    validation_steps = val_batches.n / batch_size)
