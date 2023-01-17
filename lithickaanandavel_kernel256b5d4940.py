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
import tensorflow as tf
from keras.utils import to_categorical
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Read training and test data
train_dataset = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv').values
test_dataset = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv').values
print(train_dataset.shape)
# Reshape and normalize training data
trainX = train_dataset[:, 1:].reshape(train_dataset.shape[0], 28, 28, 1).astype( 'float32' )
X_train = trainX / 255.0

y_train = train_dataset[0:,0]
y_train = to_categorical(y_train)

# Reshape and normalize test data
testX = test_dataset[:,1:].reshape(test_dataset.shape[0], 28, 28, 1).astype( 'float32' )
X_test = testX / 255.0

y_test = test_dataset[0:,0]
y_test = to_categorical(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train)
print(y_test)
# Build the CNN model
from keras import backend as k

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.Dense(units=25, activation='sigmoid'))


# Compile model
cnn.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

# Fit the model
cnn.fit(X_train, y_train,
          epochs=20,
          batch_size= 160)
score = cnn.evaluate(X_test, y_test, batch_size=64)
print(score)
cnn.summary()