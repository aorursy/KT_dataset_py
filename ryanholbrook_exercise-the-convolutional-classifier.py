# Setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.computer_vision.ex1 import *



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

import tensorflow_hub as hub



pretrained_base = tf.keras.models.load_model(

    '../input/cv-course-models/cv-course-models/inceptionv1'

)
# YOUR_CODE_HERE

pretrained_base.trainable = ____



# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
from tensorflow import keras

from tensorflow.keras import layers



model = keras.Sequential([

    pretrained_base,

    layers.Flatten(),

    # YOUR CODE HERE. Attach a head of dense layers.

    # ____

])



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
# YOUR CODE HERE: what loss function should you use for a binary

# classification problem? (Your answer for each should be a string.)

optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

model.compile(

    optimizer=optimizer,

    loss = ____,

    metrics=[____],

)



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=30,

)
import pandas as pd

history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
# View the solution (Run this code cell to receive credit!)

q_4.check()