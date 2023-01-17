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
!pip install tensorflow==2.0.0
from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
import random



plt.figure()

plt.imshow(train_images[random.randint(0,len(train_labels)-1)])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=15)
test_loss,test_accuracy = model.evaluate(test_images,test_labels,verbose=2)

print()

print('Test accuracy: ', test_accuracy)