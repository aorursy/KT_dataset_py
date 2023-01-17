# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw = pd.read_csv("../input/train.csv", sep=',')
trainLabels = raw['label']
trainInputs = raw.drop(columns = ['label'], axis=1).values/255
import tensorflow as tf
from tensorflow import keras
with tf.Session():
    trainLabels = tf.one_hot(trainLabels.astype('int32'), 10).eval()
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.tanh),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy')
model.fit(trainInputs, trainLabels, epochs=10)
test = pd.read_csv("../input/test.csv", sep=',')
test = test.values/255
output = model.predict(test)
output = np.argmax(output, axis = 1)
output = pd.DataFrame({'ImageId' : np.arange(1,len(output)+1), 'label' : output})
output.to_csv("submission2.csv", index=False)
