# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load MNIST dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Number of labels

num_labels = len(np.unique(y_train))
# Convert labels to one hot vector

y_train = tf.keras.utils.to_categorical(y_train)

y_test = tf.keras.utils.to_categorical(y_test)
# Image dimensions

image_size = X_train.shape[1]

input_size = image_size * image_size

# Resize inputs and normalize

X_train = np.reshape(X_train, [-1, input_size])

X_test = np.reshape(X_test, [-1, input_size])

X_train = X_train.astype(np.float32)/255.

X_test = X_test.astype(np.float32)/255.
# NN parameters

BATCH_SIZE = 128

HIDDEN_UNITS = 256

DROPOUT = 0.45

# Construct the model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(HIDDEN_UNITS, input_dim=input_size))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(tf.keras.layers.Dense(HIDDEN_UNITS))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(tf.keras.layers.Dense(num_labels))

model.add(tf.keras.layers.Activation('softmax')) # Output of MLP

model.summary()

tf.keras.utils.plot_model(model, to_file='/kaggle/working/mlp-mnist.png', show_shapes=True)
# Loss function , Optimizer Adam, Accuracy

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

# Train the network

model.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE)
# Validate the model

_, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print("\n Test Accuracy: %.1f%%"%(100*acc))