from sklearn import datasets

from sklearn import model_selection

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
model = tf.keras.models.Sequential([tf.keras.layers.Dense(4, activation=tf.nn.relu), tf.keras.layers.Dense(3, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000)
pred = model.predict(X_test).argmax(axis=1)

pred
(pred == y_test).sum()/len(pred)