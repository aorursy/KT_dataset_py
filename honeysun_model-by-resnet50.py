%matplotlib inline

import os, sys, time

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn

import tensorflow as tf

from tensorflow import keras



print(sys.version_info)

for module in tf, mpl, np, pd, sklearn, keras:

    print(module.__name__, module.__version__)
train_dir = "../input/10-monkey-species/training/training"

valid_dir = "../input/10-monkey-species/validation/validation"

label_file = "../input/10-monkey-species/monkey_labels.txt"

print(os.listdir(train_dir))

print(os.listdir(valid_dir))



labels = pd.read_csv(label_file, header=0)

print(labels)
height = 224

width = 224

channels = 3

batch_size = 24

num_classes = 10



train_datagen = keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function = keras.applications.resnet50.preprocess_input,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode = 'nearest',

)

train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size = (height, width),

                                                   batch_size = batch_size,

                                                   seed = 7,

                                                   shuffle = True,

                                                   class_mode = "categorical")



valid_datagen = keras.preprocessing.image.ImageDataGenerator(

    preprocessing_function = keras.applications.resnet50.preprocess_input)

valid_generator = valid_datagen.flow_from_directory(valid_dir,

                                                    target_size = (height, width),

                                                    batch_size = batch_size,

                                                    seed = 7,

                                                    shuffle = False,

                                                    class_mode = "categorical")



train_num = train_generator.samples

valid_num = valid_generator.samples

print(train_num, valid_num)
for i in range(2):

    x, y = train_generator.next()

    print(x.shape, y.shape)

    print(y)
resnet50_fine_tune = keras.models.Sequential()

resnet50_fine_tune.add(keras.applications.ResNet50(include_top = False,

                                                   pooling = 'avg',

                                                   weights = 'imagenet'))

resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation = 'softmax'))

resnet50_fine_tune.layers[0].trainable = False



resnet50_fine_tune.compile(loss="categorical_crossentropy",

                           optimizer="sgd", metrics=['accuracy'])

resnet50_fine_tune.summary()
epochs = 10

history = resnet50_fine_tune.fit_generator(train_generator,

                                           steps_per_epoch = train_num // batch_size,

                                           epochs = epochs,

                                           validation_data = valid_generator,

                                           validation_steps = valid_num // batch_size)
def plot_learning_curves(history, label, epcohs, min_value, max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_'+label]

    pd.DataFrame(data).plot(figsize=(8, 5))

    plt.grid(True)

    plt.axis([0, epochs, min_value, max_value])

    plt.show()

    

plot_learning_curves(history, 'accuracy', epochs, 0, 1)

plot_learning_curves(history, 'loss', epochs, 0, 2)