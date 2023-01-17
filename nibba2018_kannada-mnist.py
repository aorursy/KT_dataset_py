# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

cross_data = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
train_y = train_data['label']

train_X = train_data.drop(['label'], axis = 1)
import matplotlib.pyplot as plt

import tensorflow as tf
model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Dense(784, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))



model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_X.to_numpy(), train_y.to_numpy(), epochs=5)
cross_y = cross_data['label']

cross_X = cross_data.drop(['label'], axis = 1)
val_loss, val_acc = model.evaluate(cross_X.to_numpy(), cross_y.to_numpy())
test_label = test_data['id']

test_data = test_data.drop(['id'], axis =1)
test_predictions = model.predict([test_data.to_numpy()])
test_predictions_final = []

for i in range(0, len(test_predictions)):

    test_predictions_final.append(np.argmax(test_predictions[i]))
final_predictions = pd.DataFrame(

    {

        'id' : test_label,

        'label': test_predictions_final

    })
final_predictions.to_csv("submission.csv", index = False)