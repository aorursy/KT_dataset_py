# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

print(tf.__version__)
tf.test.is_gpu_available()
from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_labels.shape
test_images.shape
test_labels
import numpy as np



train_images = np.expand_dims(train_images, -1)

train_images.shape



test_images = np.expand_dims(test_images, -1)

test_images.shape
model = tf.keras.Sequential()



"""

number of filters: increases exponentially, e.g. 2^n -->拟合能力强大

ksize: (3, 3) or (5, 5) --> 经验之谈

padding (default)='valid' --> 尽量不padding

MaxPool2D (default)=(2, 2) ---> shrink its size by a factor of 2x2

input_shape=(28, 28, 1)

"""



model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_images.shape[1:], activation='relu'))

model.add(tf.keras.layers.MaxPool2D())

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))



# To obtain a 2D data that can be fed in to the fully connected layer

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
history.history.keys()
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')

plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.plot(history.epoch, history.history.get('loss'), label='loss')

plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')