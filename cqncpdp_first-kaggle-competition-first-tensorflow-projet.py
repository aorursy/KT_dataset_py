# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# getting the data
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# what does the data look like ?
test_data.head()
# spliting label and data
train_label = training_data.label
train_pixel = training_data.iloc[:,1:]
#let's try to visualize an image
plt.imshow(train_pixel.iloc[3,:].values.reshape(28,28))
# preprocess the data
train_pixel /= 255.0
test_data /= 255.0
# model - simple classification
model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# and we can start training it
model.fit(train_pixel.values, train_label.values, epochs=10)
predictions = model.predict(test_data)
type(predictions)
predic = np.argmax(predictions,axis=1)
ImageId = test_data.index + 1
my_submission = pd.DataFrame({'ImageId': ImageId, 'Label': predic})
my_submission.to_csv('submission.csv', index=False)
