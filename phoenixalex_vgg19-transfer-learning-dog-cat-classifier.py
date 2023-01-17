# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt
height = 224

width = 224

channel = 3

batch_size = 32



train_dir = '../input/cat-and-dog/training_set/training_set/'

valid_dir = '../input/cat-and-dog/test_set/test_set/'
train_data_gen = keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function=keras.applications.vgg19.preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    fill_mode='nearest',

    horizontal_flip=True

)

train_generator = train_data_gen.flow_from_directory(

    train_dir,

    target_size=(height, width),

    class_mode='binary',

    shuffle=True,

    batch_size=batch_size,

    seed=1

)

valid_data_gen = keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function=keras.applications.vgg19.preprocess_input

)

valid_generator = valid_data_gen.flow_from_directory(

    valid_dir,

    target_size=(height, width),

    class_mode='binary',

    batch_size=batch_size

)
# 下载VGG19网络

vgg19_conv_base = keras.applications.vgg19.VGG19(

    include_top=False,

    weights='imagenet'

)
vgg19_conv_base.summary()
# 搭建模型

model = keras.Sequential([

    vgg19_conv_base,

    keras.layers.GlobalAvgPool2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(1, activation='sigmoid')

])

model.summary()
model.compile(

    loss=keras.losses.binary_crossentropy,

    optimizer=keras.optimizers.Adam(lr=1e-3),

    metrics=['accuracy']

)



callbacks = [

    keras.callbacks.EarlyStopping(patience=15),

    keras.callbacks.ReduceLROnPlateau(patience=10),

    keras.callbacks.ModelCheckpoint('./vgg19_transfer_cat_dog_classifier.tf', save_best_only=True)

]

epochs = 100

train_num = train_generator.samples

valid_num = valid_generator.samples



history = model.fit(train_generator,

                    epochs=epochs,

                    steps_per_epoch=train_num // batch_size,

                    validation_data=valid_generator,

                    validation_steps=valid_num // batch_size,

                    callbacks=callbacks

                    )
history_df = pd.DataFrame(history.history)

history_df.plot(figsize=(10, 5))

plt.grid()