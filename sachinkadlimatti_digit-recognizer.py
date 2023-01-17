from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_size = (28, 28, 1)
X_train / 255

X_test / 255
MaxPooling2D

model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_size))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,kernel_size=(3,3), input_shape=input_size))

model.add(Flatten())

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(64, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train,y=y_train, epochs=30)
model.evaluate(X_test, y_test)
image_pred=model.predict_classes(X_test)

image_pred
# output file

output = pd.DataFrame({'ImageId': np.array(range(1,10001)), 'Label': image_pred})

output
## Loading the output to the CSV file for  suubmission

output.to_csv("outcome.csv", index=False)