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
import numpy as np

#import cv2

import tensorflow.keras as keras

import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt



plt.imshow(x_train[0],cmap=plt.cm.binary)

plt.show()


x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)



print(x_train[0].shape)



plt.imshow(x_train[0],cmap=plt.cm.binary)

plt.show()

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss)

print(val_acc)
y_predict = model.predict(x_test)

y_pred = []

for i in range(0,10000):

    y_pred.append(np.argmax(y_predict[i]))

#y_predict = np.argmax(y_predict[i])

cm = confusion_matrix(y_test,y_pred)



print(accuracy_score(y_test, y_pred))
