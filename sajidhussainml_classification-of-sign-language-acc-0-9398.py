# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from keras.utils import to_categorical 

from keras import backend as K

from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.layers import Dropout

from keras.layers.core import Activation





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/sign_mnist_train.csv')

dftest = pd.read_csv('../input/sign_mnist_test.csv')

print(df.shape)

df.head()

print(dftest.shape)

dftest.head()
train=df.values[0:,1:]

labels = df.values[0:,0]

labels = to_categorical(labels)

sample = train[1]

plt.imshow(sample.reshape((28,28)))

test = dftest.values[0:,1:]

testlablels = dftest.values[0:,0]

testlables = to_categorical(testlablels)
print(train.shape,labels.shape)

#normalizing the dataset

train=train/255

train=train.reshape((27455,28,28,1))

test = test/255

test 

plt.imshow(train[1].reshape((28,28)))

print(train.shape,labels.shape)
#mnist = tf.keras.datasets.fashion_mnist

#(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#training_images=train / 255.0

#test_images=test_images / 255.0

test_images = test.reshape(7172, 28, 28, 1)

test_images=test_images/255.0

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(25, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

his=model.fit(train, labels[:,1], epochs=5)

test_loss = model.evaluate(test_images, testlables[:,1])
his.history
LOC = 20

sample = train[LOC]

plt.imshow(sample.reshape((28,28)))

lbl=labels[LOC]

print(list(lbl).index(1))
sample=sample.reshape((1,28,28,1))

res=model.predict(sample)

res=list(res[0])

mx=max(res)

print(res.index(mx))