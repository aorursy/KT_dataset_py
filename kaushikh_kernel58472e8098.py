# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train_label = train.pop('label')
train = train/255

test = test/255
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(400, activation = tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(100, activation = tf.nn.relu),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation = tf.nn.softmax)

])
model.compile(optimizer = 'adam',

loss = 'sparse_categorical_crossentropy',

metrics = ['accuracy'])
batch_size = 100

training_size = len(train_label)
import math

model.fit(train.values, train_label.values, epochs =5, steps_per_epoch=math.ceil(training_size/batch_size))
test_label = model.predict(test.values)

label = np.argmax(test_label, axis =1)

label.shape
sample = pd.read_csv('../input/sample_submission.csv')
sample.head()
index = list(range(1, label.shape[0]+1))
df = pd.DataFrame({'ImageId': index, 'Label': label})

df.head()
df.to_csv('predict.csv', index=False)