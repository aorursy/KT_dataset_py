# Python imports

import random as rn



# Numerical imports

import numpy as np

import pandas as pd



# Tensorflow imports

import tensorflow as tf



# Keras imports

from tensorflow.keras.backend import sigmoid

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import mnist

from tensorflow.keras.initializers import lecun_normal

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, AlphaDropout, Conv2D, Dense, Input, Flatten

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.utils import Sequence, get_custom_objects, to_categorical



# Plotting imports

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



# Plotting settings

sns.set()

sns.set_palette("colorblind")

sns.set_style("ticks")
def build_model():

    

    # Seed randomness

    rn.seed(0)

    np.random.seed(0)

    tf.random.set_seed(0)



    # Shapes

    input_shape = (28, 28, 1)

    output_shape = 10



    # Input layer

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Flatten())

    model.add(Dense(np.prod(input_shape), activation="selu", kernel_initializer=lecun_normal(seed=0)))



    # Hidden layers

    model.add(Dense(1024, activation="selu", kernel_initializer=lecun_normal(seed=0)))

    model.add(Dense(1024, activation="selu", kernel_initializer=lecun_normal(seed=0)))

    

    # Output layer

    model.add(Dense(output_shape, activation="softmax", kernel_initializer=lecun_normal(seed=0)))



    # Optimizer and compilation

    nadam = Adam(decay=1e-6, clipvalue=0.5)

    model.compile(optimizer=nadam, loss="categorical_crossentropy", metrics=["accuracy"])

    

    return model



model = build_model()

model.summary()
%%time



# Seed randomness

rn.seed(0)

np.random.seed(0)

tf.random.set_seed(0)



# Load and split data into test/train

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"There are {len(x_train)} training images.")

print(f"There are {len(x_test)} testing images.")



# Add channels

x_train = x_train.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28, 28, 1)



# Convert class vectors to binary class matrices

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)



# Store the best model based on val_accuracy and stop when no improvements are made

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="auto",

    verbose=0,

    patience=10,

    restore_best_weights=True

)



# Build, fit, and test model

model = build_model()

history = model.fit(

    x_train,

    y_train,

    batch_size=256,  # use group norm when <= 16; use batch norm when >= 32

    epochs=100,

    verbose=0,

    validation_split=0.2,

    shuffle=False,

    callbacks=[early_stopping]

)



y1 = history.history["accuracy"]

y2 = history.history["val_accuracy"]

x = list(range(1, len(y1) + 1))

sns.lineplot(x, y1)

sns.lineplot(x, y2)



plt.title("Accuracy per epoch")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Valid"], loc="lower right")

plt.show()



score = model.evaluate(x_test, y_test, verbose=0)

print(f"Train Loss:     {history.history['loss'][-1]:.4f}")

print(f"Valid Loss:     {history.history['val_loss'][-1]:.4f}")

print(f"Test  Loss:     {score[0]:.4f}")

print()



print(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}")

print(f"Valid Accuracy: {history.history['val_accuracy'][-1]:.4f}")

print(f"Test  Accuracy: {score[1]:.4f}")

print()