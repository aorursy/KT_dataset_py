# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def get_training_data():
    train = pd.read_csv('../input/train.csv')
    print(train.head())
    print(train.describe())
    print(train.columns)
    x_train = train.Sex.values
    for i in range(x_train.shape[0]):
        if x_train[i] == "male":
            x_train[i] = 1
        else:
            x_train[i] = 0
    y_train = to_categorical(train.Survived.values)
    return x_train, y_train
x_train, y_train = get_training_data()
print(x_train)
model = keras.Sequential()
model.add(keras.layers.Dense(10, input_shape=(1,), activation='relu'))
model.add(keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, verbose=2)
gender_submission = pd.read_csv('../input/gender_submission.csv')
print(gender_submission.head())
test = pd.read_csv('../input/test.csv')