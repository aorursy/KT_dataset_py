%matplotlib inline

import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


dataset_path = "/kaggle/input/signlang/ArASL_Database_54K_Final/ArASL_Database_54K_Final"

label_path = "/kaggle/input/signlang/ArSL_Data_Labels.csv"
df = pd.read_csv(label_path)

df.head()
classes = df.Class.unique().tolist()

classes
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rescale=1.0/255)

data_generator = datagen.flow_from_directory(

        dataset_path,

        target_size=(64, 64),

        batch_size=32,

        color_mode="grayscale",

        classes = classes)
from tensorflow.keras import models

from tensorflow.keras import layers

import tensorflow as tf

with tf.device('/device:GPU:0'):

    # build a 6-layer

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

    model.add(layers.MaxPooling2D((2, 2)))



    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))



    model.add(layers.Conv2D(64, (3, 3), activation='relu'))



    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(32, activation='softmax'))



model.summary()
from tensorflow.keras.optimizers import Adam

earlystop_callback = [

            tf.keras.callbacks.EarlyStopping(

                monitor='val_loss', patience = 10, verbose=1

            ),

            tf.keras.callbacks.ModelCheckpoint('asl_cahr.h5',verbose=1,save_best_only=True)

]



with tf.device('/device:GPU:0'):

    optimizer = Adam(lr=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



    history = model.fit(

      data_generator,

      steps_per_epoch=400,

      epochs=50,callbacks=earlystop_callback)