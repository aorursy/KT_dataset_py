#Importing modules

from __future__ import absolute_import, division, print_function

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from keras.utils import to_categorical

import matplotlib.pyplot as plt
print(tf.__version__)
#from sklearn.model_selection import train_test_split

fashion_mnist_train = pd.read_csv('../input/fashion-mnist_train.csv')

fashion_mnist_test = pd.read_csv('../input/fashion-mnist_test.csv')

#train_images, train_labels, test_images, test_labels = train_test_split(fashion_mnist_train, fashion_mnist_test, test_size = 1/6, random_state = 42)

fashion_mnist_train.head()
img_rows = 28

img_cols = 28

train_images = np.array(fashion_mnist_train.iloc[:, 1:])

#train_images.head()

train_labels = to_categorical(fashion_mnist_train.iloc[:, 0])

#train_labels.head()

test_images = np.array(fashion_mnist_test.iloc[:, 1:])

test_labels = to_categorical(fashion_mnist_test.iloc[:, 0])



train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)



train_images = train_images.astype('float32')

test_images = test_images.astype('float32')



train_images /= 255

test_images /= 255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names
#input_shape = (img_rows, img_cols, 1)

#train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols)

#test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols)

train_images.shape
len(train_images)
train_labels.shape
train_labels
len(train_labels)
test_images.shape
len(test_labels)
from keras import models

from keras import layers

imput_shape = (28,28,1)

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=input_shape))

network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)