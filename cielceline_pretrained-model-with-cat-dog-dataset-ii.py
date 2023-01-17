import tensorflow as tf

import numpy as np

import glob

import os

import random

from matplotlib import pyplot as plt

%matplotlib inline
keras = tf.keras

layers = tf.keras.layers
path = glob.glob('../input/cat-and-dog/training_set/training_set/*/*.jpg')

path = path[:1000] + path[-1000:]
label = [int(p.split('/')[5]=='cats') for p in path]
def load_preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [256, 256])

    image = tf.cast(image, tf.float32)

    image = image/255

    

    return image, label
train_ds = tf.data.Dataset.from_tensor_slices((path, label))



AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)



train_ds = train_ds.shuffle(2000).repeat().batch(32)
test_path = glob.glob('../input/cat-and-dog/test_set/test_set/*/*.jpg')

test_path = test_path[:500] + test_path[-500:]

len(test_path)
test_label = [int(p.split('/')[5]=='cats') for p in test_path]



test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))

test_ds = test_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)

    

test_ds = test_ds.batch(32).repeat()
# weights 使用预先训练好的参数 /等于None

# include_top 是否包含训练好的分类器 i.e. 全连接层，否则只使用训练好的卷积基

conv_base = keras.applications.VGG16(weights='imagenet', include_top=False)
conv_base.summary()
model = keras.Sequential()

model.add(conv_base)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
conv_base.trainable = False

model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_ds, steps_per_epoch=2000//32, epochs=10, validation_data=test_ds, validation_steps=1000//32)
len(conv_base.layers)



conv_base.trainable = True



fine_tune_at = -3

for layer in conv_base.layers[:fine_tune_at]:

    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(lr=0.00005), loss='binary_crossentropy', metrics=['acc'])
initial_epochs = 10

fine_tune_epochs = 7

total = initial_epochs + fine_tune_epochs



history = model.fit(train_ds,

                    steps_per_epoch=2000//32,

                    epochs=total,

                    initial_epoch=initial_epochs,

                    validation_data=test_ds,

                    validation_steps=1000//32)