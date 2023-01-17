import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

import pandas as pd

import numpy as np

import scipy.io

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
mat = scipy.io.loadmat('../input/mnist_data.mat')

mat
Y = mat['training_labels']

X = mat['training_data']

X_test  = mat['test_data']



X.shape, Y.shape, X_test.shape
# Reshape image , Standardize , One-hot labels

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

num_classes = 10



X = X.reshape(X.shape[0], img_rows, img_cols, 1).astype('float32')/255

Y = keras.utils.to_categorical(Y, num_classes)



X.shape, Y.shape
# Train model

model = Sequential()

model.add(Conv2D(32, kernel_size = (5, 5), activation='relu', input_shape=input_shape))

#model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.summary()
# split data into training set and testing set

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# CREATE MORE IMAGES VIA DATA AUGMENTATION - by randomly rotating, scaling, and shifting images.

datagen = ImageDataGenerator(rotation_range=10,

                             zoom_range=0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1)
history = model.fit_generator(datagen.flow(X,Y, batch_size=128), # use X,Y for final step submit

                              epochs = 100, validation_data = (X,Y),

                              verbose = 2,)
# plot the accuracy and loss in each process: training and validation

def plot_(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    f, [ax1, ax2] = plt.subplots(1,2, figsize=(15, 5))

    ax1.plot(range(len(acc)), acc, label="acc")

    ax1.plot(range(len(acc)), val_acc, label="val_acc")

    ax1.set_title("Training Accuracy vs Validation Accuracy")

    ax1.legend()



    ax2.plot(range(len(loss)), loss, label="loss")

    ax2.plot(range(len(loss)), val_loss, label="val_loss")

    ax2.set_title("Training Loss vs Validation Loss")

    ax2.legend()
plot_(history)
# Predict and Submit

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')/255



Y_predict = model.predict_classes(X_test)



predict = np.column_stack((np.arange(1,(len(X_test)+1)), Y_predict))

print('save submit')

np.savetxt("submit_v10.csv", predict, fmt='%i', delimiter=",", header='Id,Category', comments='')