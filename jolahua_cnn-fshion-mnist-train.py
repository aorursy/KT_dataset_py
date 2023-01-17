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
!pip install -q tensorflow-gpu
import keras

from keras import layers

import tensorflow as tf

import matplotlib.pyplot as plt
tf.__version__

tf.test.is_gpu_available()
fashion_mnist = keras.datasets.fashion_mnist
(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()
train_image.shape
test_image.shape
train_image = np.expand_dims(train_image, -1)
train_image.shape
test_image = np.expand_dims(test_image, -1)
model = tf.keras.Sequential()    # 顺序模型

model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=train_image.shape[1:], activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc'])
model.summary()
history = model.fit(train_image, train_label, epochs=30, validation_data=(test_image, test_label))
history.history.keys()
plt.plot(history.epoch, history.history.get('acc'), label='acc')

plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')

plt.legend()
plt.plot(history.epoch, history.history.get('loss'), label='loss')

plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')

plt.legend()