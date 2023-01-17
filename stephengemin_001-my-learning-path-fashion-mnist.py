# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# TensorFlow and tf.keras

import tensorflow as tf

from keras import models

from keras import layers

from keras import datasets

from keras import callbacks

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





# Helper libraries

import numpy as np

import seaborn as sns

import pandas as pd

import random

import matplotlib.pyplot as plt

import copy

import functools

import time





train_data_import = pd.read_csv('../input/fashion-mnist_train.csv')

test_data_import = pd.read_csv('../input/fashion-mnist_test.csv')
# def to plot accuracy and loss of model training

def plot_model_training(train_results):

    epochs = range(1, len(train_results['val_loss'])+1)

    plt.plot(epochs, train_results['acc'], 'bo', label = 'Training acc')

    plt.plot(epochs, train_results['val_acc'], 'b-', label = 'Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, train_results['loss'], 'ro', label = 'Training loss')

    plt.plot(epochs, train_results['val_loss'], 'r-', label = 'Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show
train_data = train_data_import.iloc[:,1:]

train_labels = train_data_import.iloc[:,0]

test_data = train_data_import.iloc[:,1:]

test_labels = train_data_import.iloc[:,0]



#reshape for CNN

train_data = train_data.values.reshape(train_data.shape[0],28*28) / 255.

test_data = test_data.values.reshape(test_data.shape[0],28*28) / 255.



# encoding labels

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
# split training into partial_train & val

train_data, val_data, train_labels, val_labels = train_test_split(train_data,

                                                                  train_labels,

                                                                  test_size=0.1)
#Build the model

#Only using Dense layers

base = models.Sequential()

base.add(layers.Dense(8, activation='relu', input_shape=(28*28,)))

base.add(layers.Dense(16, activation='relu', input_shape=(28*28,)))

base.add(layers.Dense(10, activation='softmax'))



base.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history_base = base.fit(train_data, train_labels,

                    epochs=50, verbose=2,

                    validation_data=(val_data, val_labels))

base.save('Fashion MNIST base')

plot_model_training(history_base.history)
#Build the model

#Only using Dense layers

more_layers = models.Sequential()

more_layers.add(layers.Dense(8, activation='relu', input_shape=(28*28,)))

more_layers.add(layers.Dense(8, activation='relu'))

more_layers.add(layers.Dense(16, activation='relu'))

more_layers.add(layers.Dense(32, activation='relu'))

more_layers.add(layers.Dense(10, activation='softmax'))

more_layers.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

# model.summary()
history1 = more_layers.fit(train_data, train_labels,

                    epochs=50, verbose=2,

                    validation_data=(val_data, val_labels))

more_layers.save('Fashion MNIST 01')

plot_model_training(history1.history)
#Build the model

#Only using Dense layers

dropout = models.Sequential()

dropout.add(layers.Dense(8, activation='relu', input_shape=(28*28,)))

dropout.add(layers.Dropout(0.2))

dropout.add(layers.Dense(16, activation='relu', input_shape=(28*28,)))

dropout.add(layers.Dense(10, activation='softmax'))



dropout.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history2 = dropout.fit(train_data, train_labels,

                    epochs=50, verbose=2,

                    validation_data=(val_data, val_labels))

dropout.save('Fashion MNIST dropout')

plot_model_training(history2.history)
# base model

base.evaluate(test_data, test_labels, batch_size=32)
# model with more layers

more_layers.evaluate(test_data, test_labels, batch_size=32)
# model with dropout

dropout.evaluate(test_data, test_labels, batch_size=32)