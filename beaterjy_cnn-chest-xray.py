# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow.keras import datasets, Sequential, layers, losses, optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
print(os.listdir("../input/chest-xray-pneumonia/chest_xray/"))

print(os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/train/"))

print(os.listdir("../input/chest-xray-pneumonia/chest_xray/chest_xray/test/"))
# height and width of images

(img_h, img_w) = 150, 150

train_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/"

test_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/"

val_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val/"



EPOCHS = 10

BATCHS = 16
# prepare img data

# training generator configuration

train_dategen = ImageDataGenerator(

    rescale=1. / 255.,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

)

# testing generator configuration

test_dategen = ImageDataGenerator(

    rescale=1. / 255., 

)

# training generator

train_gen = train_dategen.flow_from_directory(

    train_dir,

    target_size=(img_h, img_w),

    batch_size=BATCHS,

    class_mode='binary'

)

# testing generator

test_gen = test_dategen.flow_from_directory(

    test_dir,

    target_size=(img_h, img_w),

    batch_size=BATCHS,

    class_mode='binary'

)

# val generator

val_gen = test_dategen.flow_from_directory(

    val_dir,

    target_size=(img_h, img_w),

    batch_size=BATCHS,

    class_mode='binary'

)
train_samples = 5216+1

test_samples= 624

val_samples = 16+1
# network

network = Sequential([

    layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),

    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),

    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation=tf.nn.relu),

    layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    

    layers.Flatten(),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(1, activation=tf.nn.sigmoid),

])

network.build(input_shape=(BATCHS, img_h, img_w, 3))

network.summary()
# optimizer and loss

network.compile(

#                 optimizer=optimizers.Adam(lr=0.0001),

                optimizer='adam',

#                 loss=losses.BinaryCrossentropy(from_logits=True),

                loss='binary_crossentropy',

               metrics=['accuracy'])
# fit network with generators

history = network.fit_generator(

    train_gen,

    steps_per_epoch=train_samples // BATCHS,

    epochs=EPOCHS,

    validation_data=val_gen,

    validation_steps=val_samples // BATCHS,

)

history.history
# testing 

scores = network.evaluate_generator(test_gen)

print(f"{network.metrics_names[1]}: {scores[1]*100}")
# save network

network.save('network.h5')