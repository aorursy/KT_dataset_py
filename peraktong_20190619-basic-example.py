# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import keras

import pandas as pd

import tensorflow as tf

import sklearn

import matplotlib.pyplot as plt

import os

import numpy as np

from PIL import Image

import random

import pathlib

from tensorflow.keras import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D



tf.enable_eager_execution()

tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# GPU information

import subprocess

import pprint



sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)



out_str = sp.communicate()

out_list = str(out_str[0]).split('\\n')



out_dict = {}



for item in out_list:

    print(item)
train_data = pd.read_csv("../input/train.csv")
# Some outputs

train_data.shape

train_data[:10]

label_names = list(train_data.columns)

#label_names[:10]
# Luckily we don't need to deal with missing data for MNIST

X = train_data[label_names[1:]]

y = train_data[label_names[0]]
# split into training and testing:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
## convert pandas to tensor, although tensorflow can take pandas input

# normalize to 0-1

img_rows, img_cols = 28, 28

num_classes =10

X_train = X_train.values.reshape([-1,28,28,1])/255

X_test = X_test.values.reshape([-1,28,28,1])/255



#y_train = keras.utils.to_categorical(y_train, num_classes)

#y_test = keras.utils.to_categorical(y_test, num_classes)

y_train = y_train.values

y_test = y_test.values



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
## Let's train them on CNN:

# https://en.wikipedia.org/wiki/AlexNet







# define a simple CNN:

# start

# I refer to the document for cifar10, which has the original Alexnet data-structure

# Credit: https://keras.io/examples/cifar10_cnn/

model = Sequential()

# Layers

# More details for using keras :

# https://github.com/keras-team/keras/blob/master/examples



model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',

                 input_shape=(img_rows,img_cols,1)))

model.add(tf.keras.layers.Activation('relu'))



model.add(tf.keras.layers.Conv2D(32, (3, 3)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(64, (3, 3)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(num_classes))

model.add(tf.keras.layers.Activation('softmax'))







# Let's train the model using adam

model.compile(loss='categorical_crossentropy',

             optimizer=tf.train.AdamOptimizer(),

              metrics=['accuracy'])

model.summary()
batch_size =128

# You can train 10 epochs or more. 10 epoch is not bad though

epochs = 100

history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_test, y_test))
## Ah, at last

test_data = pd.read_csv("../input/test.csv")

y_pred = model.predict(test_data.values.reshape([-1,28,28,1])/255,batch_size=batch_size)

y_pred = np.argmax(y_pred,axis=1)

# save

save = pd.DataFrame()



save["ImageId"] = list(range(1,len(y_pred)+1))

save["Label"] = y_pred



# In[30]:





save.to_csv("submit.csv", index=False)


