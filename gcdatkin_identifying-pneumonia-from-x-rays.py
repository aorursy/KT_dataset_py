import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras import layers
train_dir = '../input/chest-xray-pneumonia/chest_xray/train'

val_dir = '../input/chest-xray-pneumonia/chest_xray/val'

test_dir = '../input/chest-xray-pneumonia/chest_xray/test'
img_height = 128

img_width = 128

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    train_dir,

    color_mode='grayscale',

    image_size=(img_height, img_width),

    batch_size=batch_size

)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    val_dir,

    color_mode='grayscale',

    image_size=(img_height, img_width),

    batch_size=batch_size

)



test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    test_dir,

    color_mode='grayscale',

    image_size=(img_height, img_width),

    batch_size=batch_size

)
train_ds.class_names
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(np.squeeze(images[i].numpy().astype("uint8")))

        plt.title(train_ds.class_names[labels[i]])

        plt.axis("off")
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
model = tf.keras.Sequential([

    layers.experimental.preprocessing.Rescaling(1./255),

    layers.Conv2D(32, 3, activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation='relu'),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(2, activation='softmax')

])
model.compile(

    optimizer='adam',

    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),

    metrics=['accuracy']

)
epochs = 10
model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=epochs

)
model.evaluate(test_ds)