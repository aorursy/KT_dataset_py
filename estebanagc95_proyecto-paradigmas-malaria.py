# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

''' Paginas con informacion relevante https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keoras-329fbbadc5f5'''

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import random

from sklearn.model_selection import train_test_split

from tensorflow import keras

from sklearn.utils import shuffle

from keras.utils import to_categorical

from keras.layers import normalization

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

#metodo para conseguir las imagenes de los directorios

PATH = os.getcwd()

infectedCellsPath = '../input/cell_images/cell_images/Parasitized'

uninfectedCellsPath = '../input/cell_images/cell_images/Uninfected'

labels = []

images = []

infectedCells = os.listdir(infectedCellsPath)

print(infectedCells[0])

uninfectedCells = os.listdir(uninfectedCellsPath)

for img in infectedCells:

    if img != 'Thumbs.db':

        input_img = cv2.imread(infectedCellsPath+ '/' + img )

        input_img_resize = cv2.resize(input_img,(48,48))

        images.append(input_img_resize)

        labels.append(1)

plt.imshow(images[0])

plt.title('Infected Cell')

plt.show()

for img in uninfectedCells:

        if img != 'Thumbs.db':

            input_img=cv2.imread(uninfectedCellsPath + '/'+ img )

            input_img_resize = cv2.resize(input_img,(48,48))

            images.append(input_img_resize)

            labels.append(0)

plt.imshow(images[-1])

plt.title('Uinfected Cell')

plt.show()

images = np.array(images)

labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(images,labels, train_size = 0.8517, random_state = 10019)
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(48,48,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
Prueba
model.evaluate(x_test, y_test, verbose=1)