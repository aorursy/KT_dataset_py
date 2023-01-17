

# Imports

import os, warnings

import matplotlib.pyplot as plt

from matplotlib import gridspec



import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory



# Reproducability

def set_seed(seed=31415):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()



# Set Matplotlib defaults

plt.rc('figure', autolayout=True)

plt.rc('axes', labelweight='bold', labelsize='large',

       titleweight='bold', titlesize=18, titlepad=10)

plt.rc('image', cmap='magma')

warnings.filterwarnings("ignore") # to clean up output cells





# Load training and validation sets

ds_train_ = image_dataset_from_directory(

    '../input/car-or-truck/train',

    labels='inferred',

    label_mode='binary',

    image_size=[128, 128],

    interpolation='nearest',

    batch_size=64,

    shuffle=True,

)

ds_valid_ = image_dataset_from_directory(

    '../input/car-or-truck/valid',

    labels='inferred',

    label_mode='binary',

    image_size=[128, 128],

    interpolation='nearest',

    batch_size=64,

    shuffle=False,

)



# Data Pipeline

def convert_to_float(image, label):

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label



AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (

    ds_train_

    .map(convert_to_float)

    .cache()

    .prefetch(buffer_size=AUTOTUNE)

)

ds_valid = (

    ds_valid_

    .map(convert_to_float)

    .cache()

    .prefetch(buffer_size=AUTOTUNE)

)

from tensorflow import keras

from tensorflow.keras import layers



model = keras.Sequential([



    # First Convolutional Block

    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',

                  # give the input dimensions in the first layer

                  # [height, width, color channels(RGB)]

                  input_shape=[128, 128, 3]),

    layers.MaxPool2D(),



    # Second Convolutional Block

    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),

    layers.MaxPool2D(),



    # Third Convolutional Block

    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),

    layers.MaxPool2D(),



    # Classifier Head

    layers.Flatten(),

    layers.Dense(units=6, activation="relu"),

    layers.Dense(units=1, activation="sigmoid"),

])

model.summary()
model.compile(

    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),

    loss='binary_crossentropy',

    metrics=['binary_accuracy']

)



history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=40,

)

import pandas as pd



history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();