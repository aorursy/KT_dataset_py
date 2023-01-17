# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import glob

import sys

import time

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn import metrics

import gc



%matplotlib inline



data_path = '../input/training_data/training_data/*'

input_size = 120 * 320

def load_data(input_size, path):

    print("Loading training data...")

    start = time.time()



    # load training data

    X = np.empty((0, input_size))

    y = np.empty((0, 1))

    training_data = glob.glob(path)



    # if no data, exit

    if not training_data:

        print("Data not found, exit")

        sys.exit()



    for single_npz in training_data:

        with np.load(single_npz) as data:

            train = data['train']

            train_labels = data['train_labels']

        X = np.vstack((X, train))

        y = np.vstack((y, train_labels))

        

    y = (y+1)/2.

    # X, y = shuffle(X, y)

    X = np.array([img.reshape(120, 320) for img in X])

    

    batch_siz = X.shape[0]

    

    X = X.reshape(batch_siz,120,320, 1)

    print("Image array shape: ", X.shape)

    print("Label array shape: ", y.shape)



    end = time.time()

    print("Loading data duration: %.2fs" % (end - start))



    # normalize data

    X = X / 255.



    # train validation split, 7:3

    return train_test_split(X, y, test_size=0.3)



X_train, X_valid, y_train, y_valid = load_data(input_size, data_path)

from keras import layers

from keras import models

from keras import optimizers

from keras import losses

from keras.callbacks import ModelCheckpoint,EarlyStopping

model = models.Sequential()

model.add(layers.Conv2D(3, (5, 5), activation = 'relu',data_format="channels_last", input_shape = (120, 320, 1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(36, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1164, activation = 'relu'))

model.add(layers.Dense(100, activation = 'relu'))

model.add(layers.Dense(50, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

best_model_weights = './base.model'

checkpoint = ModelCheckpoint(

    best_model_weights,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)



callbacks = [checkpoint]
model.compile(loss=losses.mean_absolute_error, optimizer='sgd')

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_valid,y_valid),

    epochs = 300, 

    verbose = 1,

    callbacks=callbacks,

    batch_size = 256

)
pred = model.predict(X_train)

plt.plot(y_train[1200:1300])

plt.plot(pred[1200:1300])



plt.ylabel('some numbers')

plt.show()
#Save the model

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss)+1)



plt.plot(epochs, loss, 'b', label = "training loss")

plt.plot(epochs, val_loss, 'r', label = "validation loss")

plt.title('Training and validation loss')

plt.legend()



plt.show()