import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

import os, cv2, random

import pandas as pd

import numpy as np

tf.random.set_seed(1)

np.random.seed(1)

%matplotlib inline



TRAIN_DIR = '../input/fashionmnist/fashion-mnist_train.csv'

TEST_DIR = '../input/fashionmnist/fashion-mnist_test.csv'

IMG_SIZE = 28

BATCH_SIZE = 32

NUM_CLASSES = 10

EPOCHS = 60

BEST_ACC = 0
train_data = pd.read_csv(TRAIN_DIR)

test_data = pd.read_csv(TEST_DIR)



# How data looks

#train_data.head()

#test_data.head()



x_train = train_data.drop('label', axis=1)

y_train = train_data['label']



x_test = test_data.drop('label', axis=1)

y_test = test_data['label']



# Free some space

del train_data, test_data



x_train = x_train / 255.0

x_test = x_test / 255.0



x_train = x_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x_test = x_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

y_test = to_categorical(y_test, num_classes=NUM_CLASSES)



print('Data processed')
def plot_history(history):

    fig, ax = plt.subplots(2,1)



    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=x_train.shape[1:]))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D((2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D((2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, verbose=1, validation_data=(x_train, y_train))



plot_history(history)
model.summary()
val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2, batch_size=BATCH_SIZE)

print(f'Validation accuracy: {val_acc}')

print(f'Validation loss: {val_loss}')
# Save model

#model.save('saved_model')
# Load model

model = keras.models.load_model('best_model')