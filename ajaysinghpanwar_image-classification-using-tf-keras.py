import numpy as np

import matplotlib.pyplot as plt

import os

import PIL

import glob



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
import pathlib



dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)

data_path = pathlib.Path(data_dir)
image_count = len(list(data_path.glob('*/*.jpg')))



print(image_count)
# Tulip images

tulips = list(data_path.glob('tulips/*'))



PIL.Image.open(str(tulips[0]))
PIL.Image.open(str(tulips[15]))
# Daisy images

daisy = list(data_path.glob('daisy/*'))



PIL.Image.open(str(daisy[100]))
batch_size = 32

img_height = 180

img_width = 180
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    data_path,

    validation_split = 0.3,

    subset = "training",

    seed = 111,

    image_size = (img_height, img_width),

    batch_size = batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    data_path,

    validation_split = 0.3,

    subset = "validation",

    seed = 111,

    image_size = (img_height, img_width),

    batch_size = batch_size)
# Name of the classes of flowers present

class_names = train_ds.class_names

print(class_names)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

     ax = plt.subplot(3, 3, i + 1)

     plt.imshow(images[i].numpy().astype("uint8"))

     plt.title(class_names[labels[i]])

     plt.axis("off")
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = 5



model = Sequential([

  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

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
model.compile(optimizer = 'adam', 

             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),

             metrics = 'accuracy')
model.summary()
epochs = 10



history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']





plt.figure(figsize = (8,8))

plt.subplot(1,2,1)

plt.plot(range(epochs), acc, label = "Training Accuracy")

plt.plot(range(epochs), val_acc, label = "Validation Accuracy")

plt.legend(loc = "lower right")

plt.title("Training and Validation Accuracy")



plt.subplot(1,2,2)

plt.plot(range(epochs), loss, label = "Training Loss")

plt.plot(range(epochs), val_loss, label = "Validation Loss")

plt.legend(loc = "lower right")

plt.title("Training and Validation Loss")
data_augmentation = keras.Sequential(

  [

    layers.experimental.preprocessing.RandomFlip("horizontal", 

                                                 input_shape = (img_height, 

                                                               img_width,

                                                               3)),

    layers.experimental.preprocessing.RandomRotation(0.1),

    layers.experimental.preprocessing.RandomZoom(0.1),

  ]

)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

     augmented_images = data_augmentation(images)

     ax = plt.subplot(3, 3, i + 1)

     plt.imshow(augmented_images[5].numpy().astype('uint8'))

     plt.axis("off")
model = Sequential([

  data_augmentation,

  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(16, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(32, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Conv2D(64, 3, padding='same', activation='relu'),

  layers.MaxPooling2D(),

  layers.Dropout(0.2),

  layers.Flatten(),

  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)

])
model.compile(optimizer = 'adam',

             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),

             metrics = 'accuracy')
model.summary()
epochs = 15



history = model.fit(train_ds, validation_data = val_ds, epochs = epochs )
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']





plt.figure(figsize = (8,8))

plt.subplot(1,2,1)

plt.plot(range(epochs), acc, label = "Training Accuracy")

plt.plot(range(epochs), val_acc, label = "Validation Accuracy")

plt.legend(loc = "lower right")

plt.title("Training and Validation Accuracy")



plt.subplot(1,2,2)

plt.plot(range(epochs), loss, label = "Training Loss")

plt.plot(range(epochs), val_loss, label = "Validation Loss")

plt.legend(loc = "lower right")

plt.title("Training and Validation Loss")
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"

sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)



img = keras.preprocessing.image.load_img(

    sunflower_path, target_size=(img_height, img_width)

)

img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0) # Create a batch



predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])



print(

    "This image most likely belongs to {} with a {:.2f} percent confidence."

    .format(class_names[np.argmax(score)], 100 * np.max(score))

)