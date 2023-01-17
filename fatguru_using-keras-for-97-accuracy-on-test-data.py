# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import scipy.ndimage as sci

import numpy as np

import matplotlib.pyplot as plt

import glob

from PIL import Image

import tensorflow as tf

import os

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
!ls ..
def load_dataSet():

    data = []

    labels = []

    parasitized = glob.glob('../input/cell_images/cell_images/Parasitized/*.png')

    uninfected = glob.glob('../input/cell_images/cell_images//Uninfected/*.png')

    for imagePath in parasitized:

        image = Image.open(imagePath)

        image = image.resize((50,50))

        data.append(np.array(image))

        labels.append(0)



    for imagePath in uninfected:

        image = Image.open(imagePath)

        image = image.resize((50,50))

        data.append(np.array(image))

        labels.append(1)

    

    shuffle_sequence = np.arange(np.array(data).shape[0])

    np.random.shuffle(shuffle_sequence)

    data = np.array(data)

    labels = np.array(labels)

    print(labels.shape)

    print(data.shape)

    data = data[shuffle_sequence]

    labels = labels[shuffle_sequence]

    data = data.astype('float32') / 255

    return data, labels
data, labels = load_dataSet()
len_data = len(data)

(x_train, x_test) = data[(int)(0.1*len_data):], data[:(int)(0.1*len_data)] 

(y_train, y_test) = labels[(int)(0.1*len_data):], labels[:(int)(0.1*len_data)] 

len_train = len(x_train)

len_test = len(x_test)
y_train = keras.utils.to_categorical(y_train, 2)

y_test = keras.utils.to_categorical(y_test, 2)
model = keras.models.Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
!mkdir training

checkpoint_path = './training/cp-{epoch:04d}.ckpt'

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,

                                              save_weights_only=True,

                                              verbose=1,

                                              period=5)



model.save_weights(checkpoint_path.format(epoch=0))

model.fit(x_train, y_train,

          batch_size=50,

          epochs=20,

          verbose=1,

          callbacks=[cp_callback])
loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)