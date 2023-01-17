# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
type(train)
test=pd.read_csv('../input/test.csv')
test.shape
train_y = train.label

train_x = train.drop(["label"],axis =1)

 
train_y.head() #checking the train_y content
train_x.head() #checking the train_x content
train_x.shape
#normalisation here ; as data must be in between 0 and 1 rather 1 to 255 

import matplotlib.pyplot as plt

train_x = train_x / 255.0

train_x.shape

type(train_x)
train_x1 = train_x.values.reshape(-1,28,28,1) #reshaping



train_x1.shape
type(train_x1)
train_x2 = train_x.values.reshape(42000, 784).astype('float32')
train_x2.shape
type(train_x2)
n_classes = 10

train_y = keras.utils.to_categorical(train_y, n_classes)
train_y.shape

type(train_y)
#let's see some numbers

import matplotlib.pyplot as plt

plt.imshow(train_x1[1][:,:,0],cmap='gray')



plt.show()

plt.imshow(train_x1[7][:,:,0],cmap='gray')

plt.show()

#let's create simple deep learning model

# import tensorflow as tf 



# model = tf.keras.Sequential([

#     tf.keras.layers.Flatten(input_shape=(28,28,1)),

#     tf.keras.layers.Dense(128, activation = tf.nn.relu),

#     tf.keras.layers.Dense(10, activation = tf.nn.softmax)

    

    

# ])





model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784,)))

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))


model.summary()
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics = ['accuracy'])

#here I trained it for a while, it was improving all the time 

#but finally network gave us 0.9840 

model.fit(train_x2, train_y, epochs=20)
# test = test.values.reshape(-1,28,28,1)

test = test.values.reshape(28000, 784).astype('float32')

test = test / 255.0

# #now let's try CNN :) If it's better then we shoud go very close to 100%

# model = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

#                            input_shape=(28, 28, 1)),

#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

#     tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(128, activation=tf.nn.relu),

#     tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

# ])
# model.compile(optimizer='adam', 

#               loss='sparse_categorical_crossentropy',

#               metrics=['accuracy'])
# model.fit(train_x,train_y, epochs=10)
pred = model.predict(test)

pred[0:4]
pred = np.argmax(pred, axis=1) # we need only the best fits 
pred = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':pred })
pred.to_csv("prediction.csv",index=False)
