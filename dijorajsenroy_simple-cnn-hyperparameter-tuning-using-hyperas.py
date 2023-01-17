import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPool2D,Dropout,LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
y_train = pd.read_csv('/kaggle/input/ahdd1/Arabic Handwritten Digits Dataset CSV/csvTrainLabel 60k x 1.csv')
X_train = pd.read_csv('/kaggle/input/ahdd1/csvTrainImages 60k x 784.csv')
y_test = pd.read_csv('/kaggle/input/ahdd1/Arabic Handwritten Digits Dataset CSV/csvTestLabel 10k x 1.csv')
X_test = pd.read_csv('/kaggle/input/ahdd1/csvTestImages 10k x 784.csv')
# Normalise and reshape
X_train=X_train.values.reshape((-1,28,28,1))/255.0
X_test=X_test.values.reshape((-1,28,28,1))/255.0
# OnehotEncode y_train
y_train=to_categorical(y_train, num_classes=10)
y_train
# Split into Train and validation
X_train,X_valid,y_train, y_valid = train_test_split(X_train, y_train,
                                                     test_size=0.1,
                                                     shuffle=True)
# Data Augmentation to reduce bias
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        #rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
# Early Stopping
class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.92):
            self.model.stop_training=True
            
# LR Scheduler
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# MODEL
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same',
                activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same',
                activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same',
                activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same',
                activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout((0.4)))
    model.add(Dense(10,activation='softmax'))
    return model
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model = build_model()
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs=10
batch_size=64
# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_valid,y_valid),
                              verbose = 2, callbacks=[learning_rate_reduction])
pd.DataFrame(history.history).plot(figsize=(10,6))
y_pred = model.predict(X_test)
y_test = to_categorical(y_test,num_classes=10)
model.evaluate(X_test,y_test)
