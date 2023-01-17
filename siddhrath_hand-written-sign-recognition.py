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

import tensorflow.keras as keras

import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

X = pd.read_csv('C:/Users/sunny/Desktop/h_signs.csv').iloc[0:, 1:]

y = pd.read_csv('C:/Users/sunny/Desktop/h_signs.csv').iloc[0:,0]
X = X.values

y = y.values



x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(648, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(324, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(162, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



#train a neural network  on train datasets

model.fit(x_train, y_train, epochs=3)



#evaluate a model and print accuracy & loss for every Epoch

val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss)

print(val_acc)

for i in range(0,580):

    y_pred.append(np.argmax(y_predict[i]))





cm = confusion_matrix(y_test,y_pred)

print(accuracy_score(y_test, y_pred))
