# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import absolute_import, division, print_function



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from keras.optimizers import RMSprop





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn

from sklearn.model_selection import train_test_split



print(tf.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape
train.info()
test.info()
train.head()
test.head()
Y_train = train["label"]

# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 
X_train = X_train.values.reshape(-1,28,28)

test = test.values.reshape(-1,28,28)
X_train[0]
X_train = X_train / 255.0

test = test / 255
# Show first digit image

plt.figure()

plt.imshow(X_train[0])

plt.colorbar()

plt.grid(False)

plt.show()
# Show first digit label

Y_train[0]
# First 25 digit images and their label plots to get sense of the train data

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i], cmap=plt.cm.binary)

    plt.xlabel(Y_train.iloc[i])

plt.show()
#expand 1 more dimention as 1 for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train.shape
test = test.reshape(test.shape[0], 28, 28,1)

test.shape
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28, 1)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
# For classification problems we use sparse_categorical_crossentropy loss function 

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=5)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
test_loss, test_acc = model.evaluate(X_val, Y_val)

print('Test accuracy:', test_acc)
model = keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Dropout(0.25),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.50),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer = 'adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=30)
test_loss, test_acc = model.evaluate(X_val, Y_val)

print('Test accuracy:', test_acc)
predictions = model.predict(test)
predictions[0]
np.argmax(predictions[0])
# select the indix with the maximum probability

predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)

submission.to_csv("mnist_submission_v6.csv",index=False)
submission.head()