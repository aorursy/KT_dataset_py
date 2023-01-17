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
from __future__ import absolute_import, division, print_function, unicode_literals



# Install TensorFlow



import tensorflow as tf
import pandas as pd

mnist_test = (pd.read_csv("../input/mnist-in-csv/mnist_test.csv")).to_numpy()

mnist_train = (pd.read_csv("../input/mnist-in-csv/mnist_train.csv")).to_numpy()

x_test = mnist_test[:,1:] 

x_train = mnist_train[:,1:]



x_test = np.reshape(x_test,(x_test.shape[0],28,-1))

x_train = np.reshape(x_train,(x_train.shape[0],28,-1))



y_test = mnist_test[:,0]

y_train = mnist_train[:,0]



x_train = x_train[...,tf.newaxis]

x_test = x_test[...,tf.newaxis]



x_train = x_train / 255.0

x_test = x_test / 255.0
model1 = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28,1)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax')

])

model1.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model1_ = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28,1)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax')

])

model1_.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model2 = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28,1)),

  tf.keras.layers.MaxPooling2D(),

  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),

  tf.keras.layers.MaxPooling2D(),

  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

  tf.keras.layers.MaxPooling2D(),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation='softmax')

])





model2.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=5)

model1_.fit(x_train, y_train, epochs=20)

model2.fit(x_train, y_train, epochs=20)
model1.evaluate(x_test,  y_test, verbose=2)

model1_.evaluate(x_test,  y_test, verbose=2)

model2.evaluate(x_test,  y_test, verbose=2)
model1.save('mnist.h5')

model1_.save('mnist_.h5')

model2.save('mnistconv.h5')