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
from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D,Conv1D

from keras import backend as K

import pandas as pd



batch_size = 128

num_classes = 10

epochs = 100



# input image dimensions

img_rows, img_cols = 28, 28



# the data, split between train and test sets

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')



train_set_y = train_set.iloc[:,0].values

train_set_x =  train_set.iloc[:,1:].values



test_set_x =  test_set



x_train = train_set_x.astype('float32')

x_test = test_set_x.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(train_set_y, num_classes)

y_test = keras.utils.to_categorical(test_set_y, num_classes)



display(x_train.shape)

display(y_train.shape)
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(x_train,y_train,train_size = 0.8)
model = Sequential()

model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test_val, y_test_val))
y_test = model.predict(x_test)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

y_test = pd.DataFrame(y_test)

file_out = pd.DataFrame()

file_out["ImageId"] = y_test.index.values+1

file_out['Label'] = y_test.idxmax(axis = 1)

file_out.to_csv("Submissions.csv",index = False)