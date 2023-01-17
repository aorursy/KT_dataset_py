# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
import sys

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fashion_mnist=keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(len(x_train),len(y_train))
print(len(x_test),len(y_test))
pt_x_train = []
pt_y_train = []
pt_x_test = []
pt_y_test = []

tl_x_train = []
tl_y_train = []
tl_x_test = []
tl_y_test = []

m = 60000
for i in range(m):
    if y_train[i] < 5:
        pt_x_train.append(x_train[i] / 255)
        pt_y_train.append(y_train[i])
    else:
        tl_x_train.append(x_train[i] / 255)
        tl_y_train.append(y_train[i])
        
m2 = 10000
for i in range(m2):
    if y_test[i] < 5:
        pt_x_test.append(x_test[i] / 255)
        pt_y_test.append(x_test[i])
    else:
        tl_x_test.append(x_test[i] / 255)
        tl_y_test.append(y_test[i])
        
from keras.utils import np_utils
pt_x_train = np.asarray(pt_x_train).reshape(-1, 28, 28)
pt_x_test = np.asarray(pt_x_test).reshape(-1, 28, 28)
pt_y_train = np_utils.to_categorical(np.asarray(pt_y_train))
pt_y_test = np_utils.to_categorical(np.asarray(pt_y_test))


tl_x_train = np.asarray(tl_x_train).reshape(-1, 28, 28)
tl_x_test = np.asarray(tl_x_test).reshape(-1, 28, 28)
tl_y_train = np_utils.to_categorical(np.asarray(tl_y_train))
tl_y_test = np_utils.to_categorical(np.asarray(tl_y_test))
print(pt_x_train.shape, pt_y_train.shape)
print(pt_x_test.shape, pt_y_test.shape)

print(tl_x_train.shape, tl_y_train.shape)
print(tl_x_test.shape, tl_y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPool2D, Dropout
model = Sequential()

model.add(Conv2D(32, 5, input_shape=(28, 28 ,1)))
model.add(Activation('relu'))

model.add(Conv2D(16, 5, activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(8, 3, activation='relu'))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
import datetime
start = datetime.datetime.now()
model.fit(pt_x_train, pt_y_train,
         validation_data=(pt_x_test, pt_y_test),
         nb_epoch=10,
         shuffle=True,
         batch_size=100,
         verbose=2)

end = datetime.datetime.now()
print(end-start)
model.layers
for layer in model.layers[:6]:
    layer.trainable = False

for layer in model.layers:
    print(layer.trainable)
tl_model = Sequential(model.layers[:6])
tl_model.add(Dense(128))
tl_model.add(Activation('relu'))
tl_model.add(Dense(10))
tl_model.add(Activation('softmax'))
tl_model.summary()
tl_model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
start = datetime.datetime.now()
tl_model.fit(tl_x_train, tl_y_train,
            validation_data=(tl_x_test, tl_y_test),
            epochs=10,
            shuffle=True,
            batch_size=100,
            verbose=2)
end = datetime.datetime.now()
print(end-start)
