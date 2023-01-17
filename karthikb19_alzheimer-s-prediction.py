# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import PIL
import os
import tensorflow as tf
train_loc = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_loc,
    validation_split = 0.3,
    subset="training",
    image_size=(200, 200),
    batch_size=32,
    seed=42
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_loc, 
    validation_split = 0.3,
    subset="validation",
    image_size=(200, 200),
    batch_size=32,
    seed=42
)
# ND: No Dementia
# VMID: Very Mild Dementia
# MID: Mild Dementia
# MOD: Moderate Dementia

classes = ["ND", "VMID", "MID", "MOD"]

train_dataset.class_names = classes
valid_dataset.class_names = classes
plt.figure(figsize=(16, 16))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[labels[i]])
        plt.axis("off")
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)
moss = tf.keras.Sequential()

moss.add(tf.keras.applications.InceptionResNetV2(weights = 'imagenet', input_shape=(200, 200, 3), include_top=False))

moss.add(tf.keras.layers.Flatten())

moss.add(tf.keras.layers.Dense(128, activation="relu"))

moss.add(tf.keras.layers.Dropout(0.5))

moss.add(tf.keras.layers.Dense(4, activation="softmax"))
# Using Accuracy as my metric because I want to measure how accurately the model predicts a particular class
# AUC is also good since this is a skewed dataset

moss.compile(optimizer = tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Making certian Layers not trainable
for layer in moss.layers[:-6]:
    layer.trainable = False
history = moss.fit(train_dataset, validation_data=valid_dataset, epochs=10)
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
test_loc = "../input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test"
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_loc,
    image_size=(200, 200),
    batch_size=32
)

test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val = moss.evaluate(test_dataset)
print(val)