# Setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.computer_vision.ex6 import *



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_datasets as tfds



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

# all of the "factor" parameters indicate a percent-change

augment = keras.Sequential([

    # preprocessing.RandomContrast(factor=0.5),

    preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right

    # preprocessing.RandomFlip(mode='vertical'), # meaning, top-to-bottom

    # preprocessing.RandomWidth(factor=0.15), # horizontal stretch

    # preprocessing.RandomRotation(factor=0.20),

    # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

])





ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))



plt.figure(figsize=(10,10))

for i in range(16):

    image = augment(ex, training=True)

    plt.subplot(4, 4, i+1)

    plt.imshow(tf.squeeze(image))

    plt.axis('off')

plt.show()
ds, ds_info = tfds.load('eurosat/rgb:2.0.0',

                        with_info=True,

                        split='train',

                        data_dir='../input/eurosat',

                        download=False)

ds = ds.shuffle(1024)



tfds.show_examples(ds, ds_info);
# View the solution (Run this code cell to receive credit!)

q_1.check()
# Lines below will give you a hint 

#q_1.solution()
ds, ds_info = tfds.load('tf_flowers:3.0.1',

                        with_info=True,

                        split='train',

                        data_dir='../input/tensorflow-flowers',

                        download=False)

ds = ds.shuffle(1024)



tfds.show_examples(ds, ds_info);
# View the solution (Run this code cell to receive credit!)

q_2.check()
# Lines below will give you a hint 

#q_2.solution()
from tensorflow import keras

from tensorflow.keras import layers



model = keras.Sequential([

    layers.InputLayer(input_shape=[128, 128, 3]),

    

    # Data Augmentation

    # ____,



    # Block One

    layers.BatchNormalization(renorm=True),

    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),

    layers.MaxPool2D(),



    # Block Two

    layers.BatchNormalization(renorm=True),

    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),

    layers.MaxPool2D(),



    # Block Three

    layers.BatchNormalization(renorm=True),

    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),

    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),

    layers.MaxPool2D(),



    # Head

    layers.BatchNormalization(renorm=True),

    layers.Flatten(),

    layers.Dense(8, activation='relu'),

    layers.Dense(1, activation='sigmoid'),

])



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

model.compile(

    optimizer=optimizer,

    loss='binary_crossentropy',

    metrics=['binary_accuracy'],

)



history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=50,

)



# Plot learning curves

import pandas as pd

history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
# View the solution (Run this code cell to receive credit!)

q_4.solution()