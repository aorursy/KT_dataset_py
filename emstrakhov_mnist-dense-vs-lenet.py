# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D

from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape
train.head(3)
train['label'].value_counts()
test.shape
y_train = train["label"]

X_train = train.drop('label', axis=1)
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train.shape
y_train = to_categorical(y_train, num_classes = 10)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, 

                                                      y_train, 

                                                      test_size = 0.25, 

                                                      random_state=31)
sns.set(style='white', context='notebook', palette='deep')
g = plt.imshow(X_train[0][:,:,0])
X_train_flat = X_train.reshape(-1, 784)

X_train_flat.shape
X_valid_flat = X_valid.reshape(-1, 784)

X_valid_flat.shape
model = Sequential()

model.add(Dense(200, activation='relu', input_shape=(784,)))

# model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(X_train_flat, y_train,

                    verbose=1,

                    epochs=10,

                    validation_data=(X_valid_flat, y_valid))
model_2 = Sequential()

model_2.add(Dense(100, activation='relu', input_shape=(784,)))

model_2.add(Dense(100, activation='relu'))

model_2.add(Dense(10, activation='softmax'))



model_2.summary()
model_2.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model_2.fit(X_train_flat, y_train,

                      verbose=1,

                      epochs=10,

                      validation_data=(X_valid_flat, y_valid))
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)
history = model_2.fit(X_train_flat, y_train,

                      verbose=1,

                      epochs=20,

                      validation_data=(X_valid_flat, y_valid),

                      callbacks = [early_stopping_monitor])
# Pad images with 0s

X_train_pad = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')

X_valid_pad = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')



X_train_pad[0].shape
plt.imshow(X_train_pad[0][:,:,0])
lenet = Sequential()



lenet.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))

lenet.add(AveragePooling2D())



lenet.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

lenet.add(AveragePooling2D())



lenet.add(Flatten())



lenet.add(Dense(units=120, activation='relu'))



lenet.add(Dense(units=84, activation='relu'))



lenet.add(Dense(units=10, activation = 'softmax'))
lenet.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
lenet.fit(X_train_pad, y_train,

          verbose=1,

          epochs=10,

          validation_data=(X_valid_pad, y_valid),

#           callbacks = [early_stopping_monitor]

         )
lenet.summary()