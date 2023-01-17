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
data_train = pd.read_csv("../input/csvTrainImages 13440x1024.csv" , header = None).values
print(data_train.shape)
data_label = pd.read_csv("../input/csvTrainLabel 13440x1.csv" , header = None)
print(data_label.shape)
from __future__ import division, print_function, absolute_import


import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
import tflearn.data_utils as du
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
data_label = data_label.values.astype('int32')-1
data_label = to_categorical(data_label , 28)
print(data_label)
data_test = pd.read_csv("../input/csvTestImages 3360x1024.csv" , header = None).values
data_label_test = pd.read_csv("../input/csvTestLabel 3360x1.csv" , header = None)
data_label_test = data_label_test.values.astype('int32')-1
data_label_test = to_categorical(data_label_test , 28)
data_train = data_train/255
data_test = data_test/255
print(data_test.shape)
data_train = np.reshape(data_train , (13440 , 32 , 32 , 1))
data_test = np.reshape(data_test , (3360 , 32 , 32 , 1))
print(data_train.shape)
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten , Conv2D , MaxPooling2D
from keras.optimizers import Adam ,RMSprop 
from sklearn.model_selection import train_test_split
input_size = data_train.shape[1:]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_size))
model.add(Conv2D(64 , kernel_size = (5 , 5) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128 , activation = 'relu'))
model.add(Dense(28 , activation = 'softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= 'adam',
              metrics=['accuracy'])
model.fit(data_train, data_label,
          batch_size=256,
          epochs=50,
          verbose=1,
          validation_data=(data_test, data_label_test))
score = model.evaluate(data_test, data_label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)






