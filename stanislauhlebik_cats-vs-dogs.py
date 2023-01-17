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
from keras.preprocessing.image import ImageDataGenerator 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
train = "/kaggle/input/dogs-vs-cats/train.zip"
working = "/kaggle/working"
import zipfile
with zipfile.ZipFile(train, 'r') as zip_ref:
    zip_ref.extractall(working)
train_dir = os.path.join(working, "train")
validation_dir = os.path.join(working, "validation")

os.makedirs(os.path.join(train_dir, "dogs"))
os.makedirs(os.path.join(train_dir, "cats"))

os.makedirs(os.path.join(validation_dir, "dogs"))
os.makedirs(os.path.join(validation_dir, "cats"))
import shutil
import random

split = 0.7

for file in os.listdir(train_dir):
    if random.random() > split:
        dest_dir = validation_dir
    else:
        dest_dir = train_dir
    if file.endswith("jpg"):  
        if file.startswith("dog"):
            shutil.move(os.path.join(train_dir, file), os.path.join(dest_dir, "dogs"))
        else:
            shutil.move(os.path.join(train_dir, file), os.path.join(dest_dir, "cats"))
classes = ["cats", "dogs"]

for cl in classes:
    print("%s: %d" % (cl, len(os.listdir(os.path.join(train_dir, cl)))))

for cl in classes:
    print("%s: %d" % (cl, len(os.listdir(os.path.join(validation_dir, cl)))))
cats_dir = os.path.join(train_dir, "cats")
dogs_dir = os.path.join(train_dir, "dogs")
%matplotlib inline
cats_dir = os.path.join(train_dir, "cats")
cat_file_basename = os.listdir(cats_dir)[10]
cat_file = os.path.join(cats_dir, cat_file_basename)

plt.imshow(mpimg.imread(cat_file))
def check_empty_images(d):
    for basename in os.listdir(d):
        file = os.path.join(d, basename)
        if os.path.getsize(file) == 0:
            print("found bad")
check_empty_images(cats_dir)
check_empty_images(dogs_dir)
target_dimension = 200
batch_size = 10
train_size = 0
for c in classes:
    train_size += len(os.listdir(os.path.join(train_dir, c)))
print("train size: %d" % train_size)

validation_size = 0
for c in classes:
    validation_size += len(os.listdir(os.path.join(validation_dir, c)))
print("validation size: %d" % validation_size)
import time

def train_model(model, epochs, train=None):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5, min_delta=0.001)
    if train is None:
        train = ImageDataGenerator(rescale=1./255)
    train = train.flow_from_directory(
        train_dir,
        target_size=(target_dimension, target_dimension),
        class_mode="binary",
        batch_size=batch_size,
    )

    validation = ImageDataGenerator(rescale=1./255)
    validation = validation.flow_from_directory(
        validation_dir,
        target_size=(target_dimension, target_dimension),
        class_mode="binary",
        batch_size=batch_size,
    )

    model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
    start = time.time()
    history = model.fit_generator(
        train,
        steps_per_epoch=train_size / batch_size,
        epochs=epochs,
        validation_data=validation,
        validation_steps=validation_size / batch_size,
        #callbacks=[es],
    )
    print("training took: %f secs" % (time.time() - start))
    print("accuracy %f, validation accuracy: %f" % (history.history['accuracy'][-1], history.history['val_accuracy'][-1]))

    plt.title("Accuracy")
    plt.plot(range(0, epochs), history.history['accuracy'], label="training accuracy")
    plt.plot(range(0, epochs), history.history['val_accuracy'], label="validation accuracy")
    plt.legend()
    plt.show()

    plt.title("Loss")
    plt.plot(range(0, epochs), history.history['loss'], label="training loss")
    plt.plot(range(0, epochs), history.history['val_loss'], label="validation loss")
    plt.legend()
    plt.show()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(target_dimension, target_dimension, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
train = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)
train_model(model, epochs=80, train=train)
