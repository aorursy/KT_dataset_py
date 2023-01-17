import numpy as np

import pandas as pd

import time

import os

import keras

from keras import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt



%load_ext tensorboard.notebook

!rm -rf ./logs/
if not os.path.exists(os.path.join('weights')):

    os.mkdir(os.path.join('weights'))
test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

train_data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
y_test = test_data[['label']]

x_test = test_data.drop('label', axis=1)



y_train = train_data[['label']]

x_train = train_data.drop('label', axis=1)
x_test = x_test.astype('float32')/256

x_train = x_train.astype('float32')/256
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
model = Sequential()

model.add(Dense(10, activation='softmax', input_shape=(784,)))



model.compile(

    optimizer='adadelta', 

    loss='binary_crossentropy', 

    metrics=['accuracy']

)
log_dir = 'logs/{}'.format(time.strftime("%b_%d_%Y_%H%m%s"))

tensorboard = TensorBoard(log_dir)



model.fit(x_train, y_train, epochs=30, callbacks=[tensorboard])
model.save_weights(filepath='weights/log_reg.h5')
%tensorboard --logdir={log_dir}
test_loss, test_acc = model.evaluate(x_test, y_test)



print('Test Accuracy = ', round(test_acc, 3))
model = Sequential()

model.add(Dense(256, activation='sigmoid', input_shape=(784,)))

model.add(Dropout(0.2))

model.add(Dense(128, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))



model.compile(

    optimizer='adagrad',

    loss='binary_crossentropy',

    metrics=['accuracy']

)
log_dir = 'logs/{}'.format(time.strftime("%b_%d_%Y_%H%m%s"))

tensorboard = TensorBoard(log_dir)



model.fit(x_train, y_train, epochs=30, callbacks=[tensorboard])
model.save_weights(filepath='weights/FCNN.h5')
%tensorboard --logdir={log_dir}
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Accuracy = ', round(test_acc, 3))
x_train = x_train.values.reshape(60000, 28, 28, 1)

x_test = x_test.values.reshape(10000, 28, 28, 1)
from keras.optimizers import SGD



optimizer = SGD(lr=0.2)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',

          input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(

    optimizer=optimizer,

    loss='binary_crossentropy',

    metrics=['accuracy']

)
log_dir = 'logs/{}'.format(time.strftime("%b_%d_%Y_%H%m%s"))

tensorboard = TensorBoard(log_dir)



model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard])
model.save_weights(filepath='weights/CNN.h5')
%tensorboard --logdir={log_dir}
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Accuracy = ', round(test_acc, 3))