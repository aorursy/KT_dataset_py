# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


train = pd.read_csv("../input/without1-3/mitbih_train.csv")

test = pd.read_csv("../input/without01-03/mitbih_test.csv")
train.tail(20)
test.head(10)
x_train=train.drop("0.00E+00.88",axis=1)

y_train=train['0.00E+00.88']
model=tf.keras.Sequential([

    tf.keras.layers.Dense(10,activation='relu',input_shape=[len(x_train.keys())]),

    tf.keras.layers.Dense(20,activation='relu'),

    tf.keras.layers.Dense(8,activation='relu'),

    tf.keras.layers.Dense(1,activation='relu')

])

model.compile(optimizer='adam',loss='binary_crossentropy',

            metrics=[tf.keras.metrics.SpecificityAtSensitivity(0.8)])

model.summary()
model.fit(x_train,y_train,epochs=15,validation_split=0.1)
m = tf.keras.metrics.accuracy()

m.result().numpy()
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),

              loss=tf.keras.losses.categorical_crossentropy,

              metrics=[tf.keras.metrics.CategoricalAccuracy()])



data = x_train

labels = y_train



dataset = tf.data.Dataset.from_tensor_slices((data, labels))

dataset = dataset.batch(32)

dataset = dataset.repeat()



model.fit(dataset, epochs=10, steps_per_epoch=30)