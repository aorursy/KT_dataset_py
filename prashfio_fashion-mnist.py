

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras import regularizers

#from tensorflow.keras import optimizers

#from tensorflow.keras import initializers

from tensorflow.keras.datasets import fashion_mnist
from keras.utils import to_categorical

import pandas as pd

from sklearn.model_selection import train_test_split



data_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



x_train = np.array(data_train.iloc[:, 1:])

y_train = to_categorical(np.array(data_train.iloc[:, 0]))

#y_train = to_categorical(np.array(y_train,10))



#Here we split validation data to optimiza classifier during training

#x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=13)



#Test data

x_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))

#y_test = to_categorical(np.array(y_test,10))

y_test[1]
from tensorflow.keras import backend as K



if K.image_data_format() == 'channels_first':

    train_images = x_train.reshape(x_train.shape[0], 1, 28, 28)

    test_images = x_test.reshape(x_test.shape[0], 1, 28, 28)

    input_shape = (1, 28, 28)

else:

    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)

    test_images = x_test.reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

train_images = train_images.astype('float32')

test_images = test_images.astype('float32')

train_images /= 255

test_images /= 255
test_images[1]
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

# 64 3x3 kernels

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout to avoid overfitting

model.add(Dropout(0.25))

# Flatten the results to one dimension for passing into our final layer

model.add(Flatten())

# A hidden layer to learn with

model.add(Dense(128, activation='relu'))

# Another dropout

model.add(Dropout(0.5))

# Final categorization from 0-9 with softmax

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='RMSprop',

              metrics=['accuracy'])
history = model.fit(train_images, y_train,

                    batch_size=1000,

                    epochs=10,

                    verbose=1,

                    validation_data=(test_images, y_test))