# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
url_train = '../input/train.csv'
url_test = '../input/test.csv'
train = pd.read_csv(url_train)
test = pd.read_csv(url_test)
x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
x_train.shape
x_train.head()
y_train.shape
y_train.head()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_images = x_train/255.0
train_images = tf.reshape(train_images, [-1, 28, 28])

train_labels = y_train.ravel()
model.fit(train_images, train_labels, epochs=5, steps_per_epoch = 300)
url_sub = '../input/sample_submission.csv'
samp_sub = pd.read_csv(url_sub)
samp_sub.tail()
test_images = test/255.0
test_images = tf.reshape(test_images, [-1, 28, 28])

predictions = model.predict(test_images, steps = 1)
#predictions[0]
#np.argmax(predictions[0])
y_pred = np.argmax(predictions, axis = 1)
y_pred.shape
#y_pred[:50]
sub = pd.DataFrame()
sub['ImageID'] = test.index
sub['ImageID'] = sub['ImageID'] + 1
#sub2 = pd.DataFrame()
#sub2['Label'] = y_pred
sub['Label'] = y_pred
sub.tail()
sub.to_csv('submission.csv', index=False)
