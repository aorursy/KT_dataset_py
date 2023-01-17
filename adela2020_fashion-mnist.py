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
import tensorflow as tf

print('Tensorflow version:{}'.format(tf.__version__))

print(tf.test.is_gpu_available())
from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_labels.shape
test_images.shape
test_labels
train_images = np.expand_dims(train_images, -1)
train_images.shape
test_images = np.expand_dims(test_images, -1)
test_images.shape
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=train_images.shape[1:],activation='relu'))

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc']

             )
history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images,test_labels))
history.history.keys()
plt.plot(history.epoch, history.history.get('acc'), label='acc')

plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
model.save('fashion_mnist_model1.h5')
model1 = tf.keras.Sequential()

model1.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=train_images.shape[1:],activation='relu', padding='same'))

model1.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.MaxPool2D())

model1.add(tf.keras.layers.Dropout(0.5))

model1.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.MaxPool2D())

model1.add(tf.keras.layers.Dropout(0.5))

model1.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.MaxPool2D())

model1.add(tf.keras.layers.Dropout(0.5))

model1.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))

model1.add(tf.keras.layers.MaxPool2D())

model1.add(tf.keras.layers.Dropout(0.5))

model1.add(tf.keras.layers.GlobalAveragePooling2D())

model1.add(tf.keras.layers.Dense(256, activation='relu'))

model1.add(tf.keras.layers.Dense(10, activation='softmax'))
model1.summary()
model1.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc']

             )
history2 = model1.fit(train_images, train_labels, epochs=30, validation_data=(test_images,test_labels))
plt.plot(history2.epoch, history2.history.get('acc'), label='acc',color='red')

plt.plot(history2.epoch, history2.history.get('val_acc'), label='val_acc')
model.save('fashion_mnist_model2.h5')