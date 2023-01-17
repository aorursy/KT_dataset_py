# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np
import tensorflow as tf
import time

import matplotlib.pyplot as plt

import keras
from keras.layers import Activation, Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import pandas as pd



def deep_cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

data_f = "/kaggle/input/Kannada-MNIST/"
train = np.genfromtxt(os.path.join(data_f,"train.csv"), delimiter=',')
test = np.genfromtxt(os.path.join(data_f,"test.csv"), delimiter=',')

x_train = train[1:,1:]
y_train = train[1:,0]


# x_val = val[:,1:]
# y_val = val[:,0]
x_train = x_train.reshape((-1,28,28,1))

cnn_model = deep_cnn_model((28,28,1))
cnn_model.summary()

cnn_model.fit(x_train, to_categorical(y_train), epochs=5, batch_size=128)
_, accuracy = cnn_model.evaluate(x_train, to_categorical(y_train))
print('Train Accuracy: %.2f' % (accuracy*100))

test.shape
test = test[1:,1:]
test = test.reshape((-1,28,28,1))
preds_test = np.argmax(cnn_model.predict(test),axis=1)
# Save test predictions to file
output = pd.DataFrame({'id': list(range(test.shape[0])),
                       'label': preds_test})
output.to_csv('submission.csv', index=False)
