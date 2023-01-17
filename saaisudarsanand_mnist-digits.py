import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from keras.utils import to_categorical

import pandas as pd

import numpy as np

%matplotlib inline
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot first few images

for i in range(9):

    plt.subplot(330 + 1 + i)

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
X_train = X_train/255

X_test = X_test/255
X_train = X_train.reshape(-1,28,28,1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape = (28,28,1)))

model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()
history = model.fit(X_train, y_train,epochs = 30)
X_test = X_test.reshape(-1,28,28,1)
test_loss, test_accuracy = model.evaluate(X_test,y_test)

print('Test loss is :' + str(test_loss))

print('Test accuracy is :' + str(test_accuracy))
print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['sparse_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()