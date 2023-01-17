# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.datasets import mnist

# from keras.utils import Sequence

from keras import losses, models, optimizers

from keras.layers import (Activation, Dense, Dropout, BatchNormalization)

from keras import losses, models, optimizers

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape

x_test.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

y_train = y_train.reshape(y_train.shape[0],1)

y_test = y_test.reshape(y_test.shape[0],1)



x_train.shape

model = models.Sequential()

model.add(Dense(784, input_shape=(1,28,28,1)))

model.add(Activation("relu"))

model.add(Dense(500))

model.add(Activation("relu"))

model.add(Dense(10))

model.add(Activation('softmax'))
optimizer = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, 

                                     epsilon=None, schedule_decay=0.004)

model.compile(loss = "categorical_crossentropy", optimizer = optimizer, 

                      metrics = ["accuracy"])

model.fit(x_train, y_train, epochs=10, verbose=1, validation_data = (x_test, y_test))