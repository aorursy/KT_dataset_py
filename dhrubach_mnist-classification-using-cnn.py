# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rn

import tensorflow as tf

import keras

import warnings



from scipy.io import loadmat

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%config IPCompleter.greedy = True
warnings.filterwarnings('ignore')



tf.set_random_seed(30)
mnist = loadmat('../input/mnist-original/mnist-original.mat')
x = mnist['data'].T

y = mnist['label'][0]
print('MNIST data shape - {0}'.format(x.shape))

print('MNIST label shape - {0}'.format(y.shape))
img_height = 28

img_width = 28

channels = 1



input_shape = (img_height, img_width, channels)

num_classes = 10



epoch = 20

batch_size = 128
x_reshape = x.reshape(x.shape[0], img_height, img_width, channels)



print(x_reshape.shape)
y_encoded = keras.utils.to_categorical(y, num_classes)



print(y_encoded.shape)
idx = rn.sample(range(0, len(y_encoded)), 10)

y_random = []

for i in idx:

    y_random.append([int(x) for x in y_encoded[i]])



y_random
x_reshape = x_reshape.astype('float32')

x_reshape /= 255
x_train, x_test, y_train, y_test = train_test_split(x_reshape, y_encoded, test_size = 0.25, random_state = 0)
print('training data shape : image - {0}, label - {1}'.format(x_train.shape, y_train.shape))

print('test data shape : image - {0}, label - {1}'.format(x_test.shape, y_test.shape))
import matplotlib.pyplot as plt

%matplotlib inline



idx = rn.randint(0, x_train.shape[0])

plt.imshow(x_train[idx][:,:,0],"gray") 
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.25, random_state = 0)
print('training data shape : image - {0}, label - {1}'.format(x_train.shape, y_train.shape))

print('validation data shape : image - {0}, label - {1}'.format(x_validation.shape, y_validation.shape))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D
# model

model = Sequential()



# first conv layer

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))



# second conv layer

model.add(Conv2D(64, kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# flatten and put a fully connected layer

model.add(Flatten())

model.add(Dense(128, activation='relu')) # fully connected

model.add(Dropout(0.5))



# softmax layer

model.add(Dense(num_classes, activation='softmax'))



# model summary

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
model.fit(x_train, 

         y_train,

         batch_size = batch_size,

         epochs = epoch,

         verbose = 1,

         validation_data=(x_validation, y_validation))
model.evaluate(x_test, y_test)
print(model.metrics_names)
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),

              metrics=['accuracy'])
model.fit(x_train, 

         y_train,

         batch_size = batch_size,

         epochs = epoch,

         verbose = 1,

         validation_data=(x_validation, y_validation))
model.evaluate(x_test, y_test)