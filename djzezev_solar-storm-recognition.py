import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import glob
print(tf.__version__)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/kaggle/input/solar-storm-recognition-dataset/project1/trainimg/',
  seed=123,
  image_size=(255, 255),
  batch_size=32)
class_names = train_ds.class_names
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/kaggle/input/solar-storm-recognition-dataset/project1/testimg/',
  seed=123,
  image_size=(255, 255),
  batch_size=32)
num_classes = 2

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(255, 255, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
continuum_alpha_path = "../input/solar-storm-recognition-dataset/project1/testimg/continuum/alpha/hmi.sharp_720s.5582.20150520_000000_TAI.continuum.jpg"
test_images = [
    {
        'name' : 'continuum_alpha' ,
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/continuum/alpha/hmi.sharp_720s.5582.20150520_000000_TAI.continuum.jpg'},
    {
        'name':'continuum_beta',
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/continuum/beta/hmi.sharp_720s.4166.20140531_000000_TAI.continuum.jpg'},
    {
        'name':'continuum_betax',
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/continuum/betax/hmi.sharp_720s.5807.20150728_031200_TAI.continuum.jpg'},
    {
        'name':'magnetogram_alpha',
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/magnetogram/alpha/hmi.sharp_720s.5582.20150520_000000_TAI.magnetogram.jpg'},
    {
        'name':'magnegram_beta',
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/magnetogram/beta/hmi.sharp_720s.4166.20140531_000000_TAI.magnetogram.jpg'},
    {
        'name':'magnetogram_betax',
        'path': '../input/solar-storm-recognition-dataset/project1/testimg/magnetogram/betax/hmi.sharp_720s.5807.20150728_031200_TAI.magnetogram.jpg'},
    
]

for image in test_images:
    img = keras.preprocessing.image.load_img(
    image['path'], target_size=(255, 255))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence. --- The actual image is {} "
    .format(class_names[np.argmax(score)], 100 * np.max(score), image['name']))

