# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train_file = "../input/fashion-mnist_train.csv"
data_test_file = "../input/fashion-mnist_test.csv"

data_train = pd.read_csv(data_train_file)
data_test = pd.read_csv(data_test_file)

train_labels = data_train['label']
test_labels = data_test['label']


data_train = data_train[data_train.columns[1:]]
data_test = data_test[data_test.columns[1:]]


data_train.head()
data_train.shape
len(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
data_train = data_train / 255.0

data_test = data_test / 255.0

data_train.head()

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data_train.values, train_labels.values, epochs=5)
test_loss, test_acc = model.evaluate(data_test.values, test_labels.values)

print('Test accuracy:', test_acc)
predictions = model.predict(data_test.values)
np.argmax(predictions[3])

