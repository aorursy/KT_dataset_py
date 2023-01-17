from __future__ import absolute_import, division, print_function



import os

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import train_test_split

print(tf.__version__)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
train.columns
y_train = train['label']

x_train = train.drop('label', axis=1)
x_train = x_train / 255.0

test = test / 255.0
x_train = x_train.values.reshape(-1, 28, 28)

test = test.values.reshape(-1, 28, 28)
plt.figure()

plt.imshow(x_train[0])

plt.colorbar()

plt.grid(False)

plt.show()
y_train[0]
_x_train, _x_val, _y_train, _y_val = train_test_split(x_train, y_train,

                                                      test_size=0.2,

                                                      random_state=2)
plt.figure(figsize=(10, 10))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

    plt.xlabel(y_train.iloc[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
model.fit(_x_train, _y_train, epochs=5)
test_loss, test_acc = model.evaluate(_x_val, _y_val)

print('Test accuracy: ', test_acc)
prediction = model.predict(test)
np.argmax(prediction[0])
plt.figure()

plt.imshow(test[0])

plt.colorbar()

plt.grid(False)

plt.show()
_prediction = np.argmax(prediction, axis=1)

_prediction = pd.Series(_prediction, name='Label')
submission = pd.concat([

    pd.Series(range(1, 28001), name='ImageId'),

    _prediction

], axis=1)
submission.head()