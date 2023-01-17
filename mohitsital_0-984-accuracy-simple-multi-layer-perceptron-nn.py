# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import np_utils

import seaborn as sns

from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

import time

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras



from keras.datasets import mnist



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

# model structure : first hidden layer : # of activation unit = 512 , activation = 'relu'

# second hidden layer : # of activation unit = 512 , activation = 'relu'

# output layer : # of activation unit = 10512 , activation = 'softmax'

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

# loss function is 'categorical_crossentropy' , optimizer=RMSprop()

model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])
# model fit

batch_size = 128

epochs = 20

history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])