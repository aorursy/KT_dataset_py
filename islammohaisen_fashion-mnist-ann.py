# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import libraries

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt
print(tf.__version__)
train_data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
x = train_data.drop(['label'], axis = 1)

y = train_data['label']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =1)
# feature scaling

x_train = x_train / 255.0

x_test = x_test / 255.0
# initializng the ANN

ann = keras.models.Sequential()
# Adding the input layer and the hidden layers

ann.add(keras.layers.Flatten(input_shape=(28, 28)))
# Adding the second hidden layer

ann.add(keras.layers.Dense(128, activation='relu'))
# Adding the output layer

ann.add(keras.layers.Dense(10))
# compiling the ANN

ann.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
# Training the ANN on the Training set

ann.fit(x_train, y_train, epochs=10)
# evaluate accuracy

test_loss, test_acc = ann.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
# make predictions

probability_model = tf.keras.Sequential([ann, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)