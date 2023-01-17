# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# using Seaborne for hist
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from numpy.random import seed

sns.set(color_codes=True)
%matplotlib inline

# code for Tensorflow 2 !!!
print(tf.__version__)
assert(tf.__version__ >= '2.')
FILE_NAME = "/kaggle/input/physionet2017ecg/physionet2017.csv"

data = pd.read_csv(FILE_NAME)

# shuffle the dataframe before splitting
data = data.sample(frac=1)

data.head()
# let's try first with binary classification (normal, not normal)

data.loc[data.label != 0, "label"] = 1
data['label'].hist();
X = data.iloc[:, :2000]
Y = data.iloc[:, 2001]
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)

print('X_train: ', X_train.shape)
print('X_valid: ', X_valid.shape)

# need to trasnform to sort of image (2 dim tensor) for compatibility with input of CNN
X_train = np.expand_dims(X_train, axis = 2)

X_valid = np.expand_dims(X_valid, axis = 2)
X_train.shape
# define the model

# arcitecture inspired by keras.io

def build_model(input_shape):
    # using Keras functional API
    input_layer = keras.layers.Input(input_shape)

    # kernel size changed from 3 to 5
    conv0 = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same", activation = "relu")(input_layer)
    # conv0 = keras.layers.BatchNormalization()(conv0)
    conv0 = keras.layers.MaxPooling1D(pool_size=2)(conv0)
    
    conv1 = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(conv0)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(conv1)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(conv2)
    conv3 = keras.layers.MaxPooling1D(pool_size=2)(conv3)
    
    conv4 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(conv3)
    conv4 = keras.layers.MaxPooling1D(pool_size=2)(conv4)
    
    conv5 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(conv4)
    conv5 = keras.layers.MaxPooling1D(pool_size=2)(conv5)
    conv5 = keras.layers.Dropout(0.5)(conv5)
    
    conv6 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same", activation="relu")(conv5)
    conv6 = keras.layers.MaxPooling1D(pool_size=2)(conv6)
    
    conv7 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same", activation="relu")(conv6)
    conv7 = keras.layers.MaxPooling1D(pool_size=2)(conv7)
    conv7 = keras.layers.Dropout(0.5)(conv7)
    
    conv8 = keras.layers.Conv1D(filters=512, kernel_size=5, padding="same", activation="relu")(conv7)
    conv8 = keras.layers.MaxPooling1D(pool_size=2)(conv8)
    conv8 = keras.layers.Dropout(0.5)(conv8)
    
    conv9 = keras.layers.Conv1D(filters=512, kernel_size=5, padding="same", activation="relu")(conv8)
    
    gap = keras.layers.Flatten()(conv9)
    dense1 = keras.layers.Dense(64, activation = "relu")(gap)
    dense2 = keras.layers.Dropout(0.5)(dense1)
    dense3 = keras.layers.Dense(32, activation = "relu")(dense2)
    
    output_layer = keras.layers.Dense(1, activation="sigmoid")(dense3)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return model
seed(1234)
tf.random.set_seed(1234)


model = build_model(input_shape=(2000, 1))

# we need a smaller learning rate to have a smoother convergence
# it is really important
opt = keras.optimizers.Adam(learning_rate=0.00006)

mc = tf.keras.callbacks.ModelCheckpoint(
        'ecg5000.h5', monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')

model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
NUM_EPOCHS = 100
BATCH_SIZE = 128
VAL_SPLIT = 0.2
VERBOSE = 1

import time

t_start = time.time()

# y = X
history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, validation_split = VAL_SPLIT, verbose = VERBOSE)
# visualize loss for the training
plt.figure(figsize = (10,6))
hist_loss = history.history['loss']
hist_val_loss = history.history['val_loss']

plt.plot(hist_loss, label='Training loss')
plt.plot(hist_val_loss, label='Validation loss')

plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
# visualize accuracy for the training
plt.figure(figsize = (10,6))
hist_loss = history.history['accuracy']
hist_val_loss = history.history['val_accuracy']

plt.plot(hist_loss, label='Training accuracy')
plt.plot(hist_val_loss, label='Validation accuracy')

plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()
loss, accuracy = model.evaluate(X_valid, y_valid)