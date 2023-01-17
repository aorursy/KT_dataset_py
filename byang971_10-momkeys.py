import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import sklearn

import sys

import tensorflow as tf

import time



from tensorflow import keras



print(tf.__version__)

print(sys.version_info)

for module in np, pd, sklearn, tf, keras:

    print(module.__name__, module.__version__)
train_dir = "/kaggle/input/10-monkey-species/training/training/"

valid_dir = "/kaggle/input/10-monkey-species/validation/validation/"

label_file = "/kaggle/input/10-monkey-species/monkey_labels.txt"
labels = pd.read_csv(label_file, header=0)

print(labels[' Common Name                   '])
height = 128

width = 128



channels = 3

batch_size = 64

num_classes = 10



train_datagen = keras.preprocessing.image.ImageDataGenerator(

    #像素值 都除以255

    rescale = 1./255,

    # 图片随机旋转 (40度以内)

    rotation_range = 40,

    # 图片左右位移  20%限度以内

    width_shift_range = 0.2,

    # 图片上下位移  20%限度以内

    height_shift_range = 0.2,

    # 图像剪切强度

    shear_range = 0.2,

    # 图像缩放强度

    zoom_range = 0.2,

    # 是否水平翻转

    horizontal_flip = True,

    # 放大缩小吼， 像素填充方式

    fill_mode = 'nearest',

)



train_generator = train_datagen.flow_from_directory(train_dir,

                                                  target_size = (height, width),

                                                  batch_size = batch_size,

                                                  seed = 666,

                                                  shuffle = True,

                                                  class_mode = "categorical")



valid_datagen = keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./255,

)

valid_generator = valid_datagen.flow_from_directory(valid_dir,

                                                  target_size = (height, width),

                                                  batch_size = batch_size,

                                                  seed = 666,

                                                  shuffle = True,

                                                  class_mode = "categorical")
train_num = train_generator.samples

valid_num = valid_generator.samples
model = keras.models.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size = 3, padding='same',

                       activation = 'selu', input_shape = [width, height, channels]),

    keras.layers.Conv2D(filters=32, kernel_size = 3, 

                        padding='same', activation = 'selu'),

    keras.layers.MaxPool2D(pool_size=2),

    

    keras.layers.Conv2D(filters=64, kernel_size = 3, 

                        padding='same', activation = 'selu'),

    keras.layers.Conv2D(filters=64, kernel_size = 3, 

                        padding='same', activation = 'selu'),

    keras.layers.MaxPool2D(pool_size=2),

    

    keras.layers.Conv2D(filters=128, kernel_size = 3, 

                        padding='same', activation = 'selu'),

    keras.layers.Conv2D(filters=128, kernel_size = 3, 

                        padding='same', activation = 'selu'),

    keras.layers.MaxPool2D(pool_size=2),

    

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation = 'selu'),

    

    keras.layers.Dense(num_classes, activation = 'softmax')

])



model.compile(loss="categorical_crossentropy",

             optimizer = "adam", metrics = ['accuracy'])



model.summary()
epochs = 10

#因为数据是generator 产生的 所以不能用fit函数

history = model.fit_generator(train_generator, steps_per_epoch=train_num // batch_size,

                             epochs=epochs, validation_data=valid_generator,

                             validation_steps=valid_num//batch_size)
def plot_learning_curves(history, label, epochs, min_value, max_value):

    data = {}

    data[label] = history.history[label]

    data['val_'+label] = history.history['val_' + label]

    

    pd.DataFrame(data).plot(figsize = (8,5))

    plt.grid(True)

    plt.axis([0, epochs, min_value, max_value])

    plt.show()
plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 1.7, 5.2)