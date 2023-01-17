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
# Train DataFrame

train_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

# Test DataFrame

test_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

# Check shape of each

print('Train Shape: ', train_df.shape)

print('Test Shape: ', test_df.shape)
# X_train, normalized

X_train = train_df.iloc[:,1:].values/255.0

# y_train, labels

y_train = train_df.iloc[:,0]

# X_test, normalized

X_test= test_df.iloc[:,1:].values/255.0

# X_test, labels

y_test= test_df.iloc[:,0]

# X_train

X_train = X_train.reshape(-1, 28*28)

# X_test

X_test = X_test.reshape(-1, 28*28)
# Import TensorFlow 

import tensorflow as tf

# Define model type, Sequential

model = tf.keras.models.Sequential(name = 'Fashion_MNIST')

# Define input fully connected layer

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1], )))

# Add dropout layer

model.add(tf.keras.layers.Dropout(0.2))

# Fully connected layer

model.add(tf.keras.layers.Dense(units=265, activation='tanh' ))

# Output layer

model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))



# Model Compiling

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
# Train the model

model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Model Accuracy

print("Test accuracy: {}".format(test_accuracy))