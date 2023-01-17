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
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import os
import numpy as np
import pandas as pd
import cv2
import glob

_INPUT_FILE_NAME = "../input/creditcard.csv"


def read_and_sanitize_data():
    data = pd.read_csv(_INPUT_FILE_NAME, sep=',')
    return data


def model(input_shape):
    print(input_shape)
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model



input_data = read_and_sanitize_data()
print(input_data.shape)
train_x = input_data.drop(labels='Class', axis=1)
print(train_x.shape)
train_y = input_data[['Class']]
print(train_y.shape)
train_x.drop(train_x.head(1).index, inplace=True)
train_y.drop(train_y.head(1).index, inplace=True)
model = model(train_x.shape[1:])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
train_x, train_y = shuffle(train_x, train_y, random_state=2)
history = model.fit(np.asarray(train_x), np.asarray(train_y), epochs=100, batch_size=train_x.shape[0],
                    class_weight={0: 1, 1: 100})

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
