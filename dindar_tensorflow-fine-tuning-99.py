import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import PIL
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
import random
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import save
from numpy import load

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
training_path = "../input/10-monkey-species/training/training"
testing_path = "../input/10-monkey-species/validation/validation"
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                               zoom_range = 0.2,
                               width_shift_range=0.15,
                               height_shift_range=0.15,
                               horizontal_flip=True)
training_instances = generator.flow_from_directory(training_path, 
                                                   target_size=(299, 299),
                                                   batch_size=256)

generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_instances = generator.flow_from_directory(testing_path,
                                               target_size=(299, 299),
                                               batch_size=32)
base_model = tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(299,299,3))
base_model.trainable = False
x = tf.keras.layers.GlobalAvgPool2D()(base_model.output)

x = tf.keras.layers.Dense(512, activation="relu") (x)
x = tf.keras.layers.Dense(256, activation="relu") (x)

output_layer = tf.keras.layers.Dense(10, activation="softmax") (x)
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
model.summary()
initial_epochs = 17
%%time
Batch_size=256
model_fit=model.fit(training_instances, 
                    steps_per_epoch=1098//Batch_size,
                    epochs=initial_epochs,
                    validation_data=test_instances
                   )
model.evaluate(test_instances)
base_model.trainable = True
len(base_model.layers)
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 600

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
model.summary()
len(model.trainable_variables)
%%time
fine_tune_epochs = 20
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(training_instances,
                         epochs=total_epochs,
                         initial_epoch =  model_fit.epoch[-1],
                         validation_data=test_instances)
model.evaluate(test_instances)

