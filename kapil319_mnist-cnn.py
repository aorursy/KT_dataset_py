# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

# fix dimension ordering issue

from keras import backend as K

K.set_image_dim_ordering('th')
np.random.seed(7)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][channels][width][height]

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255
# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
# define a simple CNN model

def baseline_model():

	# create model

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
model = baseline_model()

# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test, verbose=0)

scores[1]*100