from IPython.display import Image

Image('fashion.jpeg')
!pip install tensorflow-gpu==2.0.0.alpha0
tf.__version__
import numpy as np

import datetime

import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
#Loading the Fashion Mnist dataset

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28*28)
X_train.shape
#Reshape the testing subset in the same way

X_test = X_test.reshape(-1, 28*28)
X_test.shape
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
model_json = model.to_json()

with open("fashion_model.json", "w") as json_file:

    json_file.write(model_json)
model.save_weights("fashion_model.h5")
model2 = tf.keras.models.Sequential()

model2.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(784, )))

model2.add(tf.keras.layers.Dropout(0.6))

model2.add(tf.keras.layers.Dense(units=64, activation='relu'))

model2.add(tf.keras.layers.Dropout(0.4))

model2.add(tf.keras.layers.Dense(units=128, activation='relu'))

model2.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model2.summary()
model2.fit(X_train, y_train, epochs=10)
test_loss, test_accuracy = model2.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
model_json2 = model2.to_json()

with open("fashion_model2.json", "w") as json_file:

    json_file.write(model_json2)
model2.save_weights("fashion_model2.h5")
model3 = tf.keras.models.Sequential()

model3.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

model3.add(tf.keras.layers.Dropout(0.2))

model3.add(tf.keras.layers.Dense(units=256, activation='relu'))

model3.add(tf.keras.layers.Dropout(0.2))

model3.add(tf.keras.layers.Dense(units=128, activation='relu'))

model3.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model3.summary()
model3.fit(X_train, y_train, epochs=10)
test_loss, test_accuracy = model3.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
model_json3 = model3.to_json()

with open("fashion_model3.json", "w") as json_file:

    json_file.write(model_json3)
model3.save_weights("fashion_model3.h5")