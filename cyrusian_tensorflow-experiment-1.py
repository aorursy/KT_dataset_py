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
#https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
import tensorflow.keras as keras #importing deep learning library
import tensorflow as tf #deep learning library
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

import pandas as pd
df = pd.read_csv('../input/train.csv')
df.head()
df.describe()
print(df.values[0])
# print(x_train[0])
import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
y_train_2 = df['label']
df = df / 255
import numpy as np
labels = df.columns.tolist()
labels.remove('label')
df = df[labels]
df.head()
x_train_2 = df

# x_train_2.values[0]
y_train_2.values[0]
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)
model.fit(x_train_2.values, y_train_2.values, epochs=3)
# model.save('epic_num_reader.model')
# new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = model.predict(x_test)
import numpy as np

print(np.argmax(predictions[5]))
plt.imshow(x_test[5],cmap=plt.cm.binary)
plt.show()