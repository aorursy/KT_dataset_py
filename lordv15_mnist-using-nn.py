import tensorflow as tf

import keras

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



mnist = tf.keras.datasets.mnist



(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0



model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=5)

y_new = model.predict_classes(x_test)

print("Accuracy is: ",np.mean(y_new == y_test) *100)