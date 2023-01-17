! pip install tensorflow-gpu==2.0.0
import numpy as np

import tensorflow as tf

import datetime

from keras.datasets import fashion_mnist

tf.__version__
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
x_train = x_train / 255.0

x_test = x_test/255.0
x_train = x_train.reshape(-1, 28*28)

x_test = x_test.reshape(-1, 28*28)

print(x_train.shape)

print(x_test.shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 120, activation = 'relu', input_shape = (784,)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['sparse_categorical_accuracy'])
model.summary()

if (tf.test.is_gpu_available):

    print("GPU")

else:

    print("CPU")

from tensorflow.python.client import device_lib



device_lib.list_local_devices()
with tf.device("/device:GPU:0"):

    model.fit(x_train, y_train, epochs = 10)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
"""

Saving model's topology



model_json = model.to_json()

with open("fashion_model.json", "w") as json_file:

    json_file.write(model_json)



Saving model's weights

model.save_weights("fashion_model.h5")



"""