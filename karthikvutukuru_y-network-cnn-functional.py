# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Flatten, concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical, plot_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# from sparse label to categorical

num_labels = len(np.unique(y_train))

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# reshape and normalize input images

image_size = x_train.shape[1]

x_train = np.reshape(x_train,[-1, image_size, image_size, 1])

x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255
# network parameters

input_shape = (image_size, image_size, 1)

batch_size = 32

kernel_size = 3

dropout = 0.4

n_filters = 32
# Left Branch of Y network

left_inputs = Input(shape=input_shape)

x = left_inputs

filters = n_filters

# 3 layers of Conv2D-Dropout-MaxPooling2D

# number of filters doubles after each layer (32-64-128)

for i in range(3):

    X = Conv2D(filters=filters,

              kernel_size=kernel_size,

              padding='same',

              activation='relu')(x)

    X = Dropout(dropout)(X)

    X = MaxPooling2D()(X)

    filters *= 2

# Right Branch of Y Network

right_inputs = Input(shape=input_shape)

y = right_inputs

filters = n_filters

# 3 layers of Conv2D-Dropout-MaxPooling2D

# number of filters doubles after each layer (32-64-128)

for i in range(3):

    Y = Conv2D(filters=filters,

              kernel_size=kernel_size,

              padding='same',

              activation='relu')(y)

    Y = Dropout(dropout)(Y)

    Y = MaxPooling2D()(Y)

    filters *= 2
# Merge left and Right outputs from above

y = concatenate([X, Y])

# Feature Maps to Vector

y = Flatten()(y)

# Dropout

y = Dropout(dropout)(y)

outputs = Dense(num_labels, activation='softmax')(y)



# Build the model

model = Model([left_inputs, right_inputs], outputs)

model.summary()
# Graph the model

plot_model(model, to_file='/kaggle/working/y_model_cnn.png', show_shapes=True)
model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
# Train the model

model.fit([x_train, x_train], y_train, validation_data=([x_test, x_test], y_test),

         epochs=20, batch_size=batch_size )
# Determine model accuracy

score = model.evaluate([x_test, x_test],

y_test,

batch_size=batch_size,

verbose=0)

print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))