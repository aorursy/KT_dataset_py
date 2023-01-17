import numpy as np

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import os

import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras

import cv2
image_string = tf.io.read_file("../input/mnistasjpg/testSet/testSet/img_10.jpg")

image=tf.image.decode_jpeg(image_string,channels=3)
image.shape, image.numpy().max()
fig = plt.figure()

plt.subplot(1,2,1)

plt.title('Original image')

plt.imshow(image)
batch_size = 200

img_height = 28

img_width = 28

epochs = 10

train_dir = "../input/mnistasjpg/trainingSet/trainingSet/"

test_dir = "../input/mnistasjpg/trainingSample/trainingSample/"
# data_dir = tf.keras.utils.get_file(origin="../input/mnistasjpg/null", 

#                                    fname='flower_photos', 

#                                    untar=True)



# _URL

# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.3) # Generator for our training data

train_ds =  train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(img_height, img_width),

                                                           class_mode='categorical',

                                                            subset='training')





test_ds =  train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(img_height, img_width),

                                                           class_mode='categorical',

                                                            subset='validation')
# Seed value

seed_value= 0



import os

os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



import numpy as np

np.random.seed(seed_value)



import tensorflow as tf

tf.random.set_seed(seed_value)
import math

def scheduler(epoch):

  epoch_limit = 3



  if epoch < epoch_limit:

    return 0.01

  else:

    return  max(0.001 * math.exp(0.001 * (epoch_limit - epoch)) , 0.0001)







lrcallback = tf.keras.callbacks.LearningRateScheduler(scheduler)



import os, datetime



logdir = os.path.join("logs3", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)





# {epoch:02d}-{val_loss:.2f}

checkpoint_filepath = './{epoch:02d}-{val_accuracy:.2f}.checkpoint2.hdf5'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(

                              filepath=checkpoint_filepath,

                              monitor='val_accuracy',

                              mode='max',

                              save_best_only=True)







# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

#

# Multi models

#
import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
inputs = keras.Input(shape=(28, 28, 3),name="img")

x1 = layers.Conv2D(4, 5, activation="relu")(inputs)

x2 = layers.Conv2D(8, 5, activation="relu")(x1)

x3 = layers.MaxPooling2D(3)(x2)





x4 = layers.Conv2D(16, 5, activation="relu", padding='same')(x3)

x5 = layers.Conv2D(32, 5, activation="relu", padding='same')(x4)

x6 = layers.MaxPooling2D(3)(x5)





flatten = layers.Flatten()



block_3_output = layers.Concatenate()([flatten(inputs), flatten(x3), flatten(x6)])



x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(block_3_output)

x = layers.Dropout(0.05)(x)

outputs = layers.Dense(10, activation="softmax")(x)



model = keras.Model(inputs, outputs, name="mnist")

model.summary()
ls
model.compile(optimizer='adam',

              loss=tf.keras.losses.CategoricalCrossentropy(),

              metrics=['accuracy'])
# keras.backend.set_value(model.optimizer.learning_rate, 0.0001)
history = model.fit(

    train_ds,

    steps_per_epoch=len(train_ds.filepaths) // batch_size,

    epochs=epochs,

    verbose=1,

    validation_data=test_ds,

    validation_steps=len(test_ds.filepaths) // batch_size

)