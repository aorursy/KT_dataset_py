import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from os.path import join

import random

import pathlib

import tensorflow as tf

tf.enable_eager_execution()

import IPython.display as display



tf.test.is_gpu_available()
tf.random.set_random_seed(42)

np.random.seed(42)
basedir = '../input/spoonvsfork/spoon-vs-fork/spoon-vs-fork'

fork_dir = join(basedir, 'fork')

spoon_dir = join(basedir, 'spoon')

spoon_paths = [join(spoon_dir, img_path) for img_path in os.listdir(spoon_dir)]

fork_paths = [join(fork_dir, img_path) for img_path in os.listdir(fork_dir)]

img_paths = spoon_paths + fork_paths

len(img_paths)
def load_data(basedir):

    folders = os.listdir(basedir)

    print(folders)

    result = pd.DataFrame(columns=['filename', 'class'])

    for folder in folders:

        files = [join(basedir, folder, file) for file in os.listdir(join(basedir, folder))]

        df = pd.DataFrame({'filename': files, 'class': folder})

        result = pd.concat([result, df])

    return result



image_df = load_data(basedir)
def validate_data(image_df):

    result = image_df.copy()

    allowed_extensions = ['jpg', 'jpeg', 'png', 'gif']

    for img in image_df.filename:

        extension = str.lower(os.path.splitext(img)[1])[1:]

        if extension not in allowed_extensions:

                    result = result[result.filename != img]

                    print("Removed file with extension '{}'".format(extension))

    return result



image_df = validate_data(image_df)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_df.filename, image_df['class'], test_size=0.2, random_state=42)
from tensorflow.python.keras.applications import ResNet50

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

resnet = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



resnet_model = Sequential()

resnet_model.add(resnet)

resnet_model.add(Dense(2, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

resnet_model.layers[0].trainable = False
resnet_model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
batch_size=16

gen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    horizontal_flip = True,

    vertical_flip = True,

    width_shift_range = 0.1,

    height_shift_range = 0.1,

    zoom_range = 0.1,

    rotation_range = 10

    )

train_gen = gen.flow_from_dataframe(pd.DataFrame({'filename': X_train, 'class': y_train}), batch_size=batch_size)

validation_gen = gen.flow_from_dataframe(pd.DataFrame({'filename': X_test, 'class': y_test}), batch_size=batch_size)

resnet_model.fit_generator(train_gen, validation_data=validation_gen)
resnet_model.fit_generator(train_gen, validation_data=validation_gen, epochs=8)