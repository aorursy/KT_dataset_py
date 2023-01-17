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
from keras.datasets import mnist

from keras.utils import to_categorical

np_load_old = np.load

# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# restore np.load for future normal usage

np.load = np_load_old
train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))

test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
from keras import layers

from keras import models

model = models.Sequential()

# Conv2D(output_depth, (window_height, window_width))

# output_depth = 32: using 32 filters

# each of the 32 output channel contains a 26*26 grid of values (response map)

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))  # downsample

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',

loss='categorical_crossentropy',

metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)

test_acc