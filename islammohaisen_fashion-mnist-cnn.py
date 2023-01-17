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
# import libraries

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt
print(tf.__version__)
train_data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
x, y = data_prep(train_data)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=1)
# initializng the CNN

cnn = keras.models.Sequential()
# step 1: Convolution .. classicals architecture (f =32)

cnn.add(keras.layers.Conv2D(filters = 32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))
# step 2: MaxPooling

cnn.add(keras.layers.MaxPool2D(pool_size = 2, strides=2))
# step 3: Flattering

cnn.add(keras.layers.Flatten())
# step 4: Full connection

cnn.add(keras.layers.Dense(128, activation='relu'))
# step 5: output layer

cnn.add(keras.layers.Dense(10,activation='softmax'))
# compiling the CNN

cnn.compile(  optimizer='Adam', loss = tf.keras.losses.categorical_crossentropy,

              metrics=['accuracy'])
# Training the CNN on the Training and evaluating it on the Test set

history = cnn.fit(x_train, y_train,epochs=10) 
# evaluate accuracy

test_loss, test_acc = cnn.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
# make predictions

probability_model = tf.keras.Sequential([cnn, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)