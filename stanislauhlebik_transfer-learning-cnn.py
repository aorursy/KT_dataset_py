# https://www.tensorflow.org/tutorials/images/transfer_learning
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import tensorflow_datasets to fetch data, we need to use pip to install it first
# Note that it requires enabling the internet connection (see "Settings")
!pip install tensorflow_datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
(raw_train, raw_test, raw_validation), metadata = tfds.load("cats_vs_dogs", split=["train[:80%]", "train[80%:90%]", "train[90%:]"], with_info=True)
metadata
print(raw_train)
IMG_SIZE = 160
i = 0
plt.figure(figsize=(10, 10))
for img in raw_train.take(2):
    i += 1
    plt.subplot(2, 2, i)
    plt.title(metadata.features['label'].int2str(img["label"]) + " original")
    plt.imshow(img["image"])
    
    i += 1
    plt.subplot(2, 2, i)
    resized = tf.image.resize(img["image"], [IMG_SIZE, IMG_SIZE])
    plt.title(metadata.features['label'].int2str(img["label"]) + " resized")
    plt.imshow(resized.numpy().astype(np.uint8))

def format_example(sample):
    image, label = sample["image"], sample["label"]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return tf.image.resize(image, [IMG_SIZE, IMG_SIZE]), label
for img in raw_train.take(1):
    img["image"].numpy()
    formatted = format_example(img)
raw_train = raw_train.map(format_example)
raw_test = raw_test.map(format_example)
raw_validation = raw_validation.map(format_example)
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
raw_train = raw_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
raw_test = raw_test.batch(BATCH_SIZE)
raw_validation = raw_validation.batch(BATCH_SIZE)
# take(1) returns a single batches (i.e. BATCH_SIZE samples)
for image_batch, labels in raw_train.take(1):
    print(image_batch.shape)
    print(labels.shape)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
full_base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
base_model.summary()
full_base_model.summary()
converted_batch = base_model(image_batch)
print(converted_batch.shape)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(converted_batch)
print(converted_batch.shape)
print(feature_batch_average.shape)
converted_batch[0]
feature_batch_average[0]
prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
predicted = prediction_layer(feature_batch_average)
print(predicted)
print("Max probability: %f, min probability %f" % (max(predicted), min(predicted)))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer,
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
# What is trainable variables?
len(model.trainable_variables)
initial_epochs = 10
validation_steps=20

loss0, accuracy0 = model.evaluate(raw_validation, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(raw_train,
                    epochs=initial_epochs,
                    validation_data=raw_validation)
plt.subplot(211)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(212)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
base_model.trainable = True
print(len(base_model.layers))

fine_tune_from = 100
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.00001), metrics=["accuracy"])
model.summary()
len(model.trainable_variables)
history.epoch[-1]
history_fine_tune = model.fit(raw_train, epochs=initial_epochs + 10, initial_epoch=history.epoch[-1], validation_data=raw_validation)
accuracy = history.history["accuracy"] + history_fine_tune.history["accuracy"]
val_accuracy = history.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]

loss = history.history["loss"] + history_fine_tune.history["loss"]
val_loss = history.history["val_loss"] + history_fine_tune.history["val_loss"]

plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.title("Accuracy")
plt.plot(accuracy, label="accuracy")
plt.plot(val_accuracy, label="val_accuracy")
plt.plot([initial_epochs, initial_epochs], plt.ylim())
plt.legend()

plt.subplot(212)
plt.title("Loss")
plt.plot(loss, label="loss")
plt.plot(val_loss, label="val_loss")
plt.plot([initial_epochs, initial_epochs], plt.ylim())
plt.legend()