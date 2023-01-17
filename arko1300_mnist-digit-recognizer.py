import tensorflow as tf

print(tf.__version__)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[9])
x_train.shape
x_test.shape
x_train = x_train/255.0

x_test = x_test/255.0
x_train = x_train.reshape(60000, 28, 28, 1)

x_test = x_test.reshape(10000, 28, 28, 1)
x_train[0].shape
input_shape = x_train[0].shape
model = Sequential()

model.add(Conv2D(filters=32, kernel_size= (3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, kernel_size= (3, 3), activation='relu'))

model.add(MaxPool2D(2, 2))



model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,

                    validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
history.history
def plot_learningcurve(history, epochs):

    epoch_range = range(1, epochs+1)

    plt.plot(epoch_range, history.history['accuracy'])

    plt.plot(epoch_range, history.history['val_accuracy'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()



    plt.plot(epoch_range, history.history['loss'])

    plt.plot(epoch_range, history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Test', 'Val'], loc='upper left')

    plt.show()
plot_learningcurve(history, 10)