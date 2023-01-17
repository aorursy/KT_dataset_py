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

#         print(os.path.join(dirname, filename))

        pass





# Any results you write to the current directory are saved as output.
import os 

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import pandas as pd

train_dir = "../input/10-monkey-species/training/training"

validation_dir = "../input/10-monkey-species/validation/validation"

label_file = "../input/10-monkey-species/monkey_labels.txt"

print(os.path.exists(train_dir))

print(os.path.exists(validation_dir))

labels = pd.read_csv(label_file)

print(labels)
height = 128

width = 128

channels = 3

batch_size = 64

num_classes = 10
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,

                                                          rotation_range=40,

                                                          width_shift_range=0.2,

                                                          height_shift_range=0.2,

                                                          horizontal_flip=True,

                                                          fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=batch_size,

                                                    target_size=(width,height),

                                                    seed=7,

                                                    shuffle=True,

                                                    class_mode="categorical")

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(validation_dir,

                                                    batch_size=batch_size,

                                                    target_size=(width,height),

                                                    seed=7,

                                                    shuffle=True,

                                                    class_mode="categorical")

train_num = train_generator.samples

valid_num = valid_generator.samples

print(train_num, valid_num)
layers_channels = [32,64,128]

def conv_layers(model,layers_channels,activation="relu"):

    for num,channel in enumerate(layers_channels):

        if num==0: #表示第一层

            model.add(keras.layers.Conv2D(filters=channel, kernel_size=3, padding="same",activation=activation, input_shape=[width, height, channels]))

        else:

            model.add(keras.layers.Conv2D(filters=channel, kernel_size=3, padding="same",activation=activation))

        model.add(keras.layers.Conv2D(filters=channel, kernel_size=3, padding="same",activation=activation))

        model.add(keras.layers.MaxPool2D(pool_size=2))

    return model

            

model = keras.models.Sequential()

model = conv_layers(model,layers_channels)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation="relu"))

model.add(keras.layers.Dense(10,activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
epochs = 10

history = model.fit_generator(train_generator,

                              steps_per_epoch = train_num // batch_size,

                              epochs=epochs,

                              validation_data=valid_generator,

                              validation_steps=valid_num // batch_size)

def plot_learning_curves(history, label, epochs, min_value, max_value):

    data = {}

    data[label] = history.history[label]

    data["val_"+label] = history.history["val_"+label]

    pd.DataFrame(data).plot(figsize=(8,5))

    plt.grid(True)

    plt.axis([0,epochs,min_value,max_value])

    plt.show()



plot_learning_curves(history, "accuracy", epochs,0,1)

plot_learning_curves(history, "loss", epochs,1., 2.5)