# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/fashion-mnist_train.csv")

test_data = pd.read_csv("../input/fashion-mnist_test.csv")
train_data.head()
test_data.head()
train_labels = train_data.iloc[:,0:1]

test_labels = test_data.iloc[:,0:1]



train_data = train_data.drop("label", axis=1)

test_data = test_data.drop("label", axis=1)
print(train_data.shape, test_labels.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()

plt.imshow(train_data.iloc[1:2,:].values.reshape(28,28))

plt.grid(False)

plt.xlabel(class_names[train_labels.iloc[1][0]])
# Scaling

train_data = train_data / 255.0

test_data = test_data / 255.0
train_data = train_data.values.reshape(60000,28,28)

test_data = test_data.values.reshape(10000,28,28)
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer=tf.train.AdamOptimizer(), 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_data, test_labels)



print('Test accuracy:', test_acc)
predictions = model.predict(test_data)

print(predictions[0])     # Prints the confidence level for each class
print(class_names[np.argmax(predictions[0])])
print(class_names[test_labels.iloc[0][0]])