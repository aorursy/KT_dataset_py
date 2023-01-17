# - Generic -



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings, time, pathlib

warnings.filterwarnings("ignore")



# - TensorFlow - 

import tensorflow as tf

import matplotlib.pylab as plt

from tensorflow.keras import layers

from tensorflow import keras
print("TensorFlow Version Used: ",tf.__version__)
# - Load Dataset -

ds_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'



img_dir = tf.keras.utils.get_file(origin=ds_url, fname='flower_photos', untar=True)

img_dir = pathlib.Path(img_dir)



# Image Count

print("\n Total Images Downloaded: ", len(list(img_dir.glob('*/*.jpg'))))
# Parameter Definition

batch_size = 32

image_size = (300,300)

epochs = 10
''' Training - 90%'''

''' Validation - 10%'''



# Training Dataset

print(" --Training Dataset-- ")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(img_dir,validation_split=0.1,

                                                               subset="training",seed=123, image_size=image_size, batch_size=batch_size)



# Validation Dataset

print("\n --Validation Dataset-- ")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(img_dir, validation_split=0.1,

                                                             subset="validation",seed=123, image_size=image_size, batch_size=batch_size)





# Training Class Number

num_class = len(train_ds.class_names)
# Dataset Performance Config



AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Custom Function to Build Model



def build_model():

    return tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255),

                                layers.Conv2D(32, 3, activation='relu'),

                                layers.MaxPooling2D(),

                                layers.Conv2D(32, 3, activation='relu'),

                                layers.MaxPooling2D(),

                                layers.Conv2D(32, 3, activation='relu'),

                                layers.MaxPooling2D(),

                                layers.Flatten(),

                                layers.Dense(128, activation='relu'),

                                layers.Dense(num_class) ])





# Create Model

model = build_model()
# Define Optimizer List

optimizers = [ 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD' ]



opt_res = []



# Compile & Train

for optimizer in optimizers:

    model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    print(f"Fitting the model with {optimizer}")

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=-1)

    gc.collect()  #Free Memory after each run

    opt_res.append(history.history['accuracy'])

fully_nested = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in opt_res]

names = ['sublist%d'%(i+1) for i in range(len(fully_nested))]



fig = plt.figure(figsize=(15,10))



for l in fully_nested:

    plt.plot(*l)



plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(optimizers, fontsize=9, loc = 'upper right', bbox_to_anchor=(1.1, 1.01))

plt.title("Optimizer Performance Comparison", fontsize=25)

plt.show()