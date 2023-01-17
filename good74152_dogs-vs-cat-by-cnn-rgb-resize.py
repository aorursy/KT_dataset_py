#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
from __future__ import absolute_import, division, print_function, unicode_literals



import os

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
zip_dir_base = os.path.dirname(zip_dir)

!find $zip_dir_base -type d -print
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')



train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
num_cats_tr = len(os.listdir(train_cats_dir))

num_dogs_tr = len(os.listdir(train_dogs_dir))



num_cats_val = len(os.listdir(validation_cats_dir))

num_dogs_val = len(os.listdir(validation_dogs_dir))



total_train = num_cats_tr + num_dogs_tr

total_val = num_cats_val + num_dogs_val
print('total training cat images:', num_cats_tr)

print('total training dog images:', num_dogs_tr)



print('total validation cat images:', num_cats_val)

print('total validation dog images:', num_dogs_val)

print("--")

print("Total training images:", total_train)

print("Total validation images:", total_val)
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables

IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels
train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE, 

                                                           directory=train_dir, 

                                                           shuffle=True, 

                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150) 

                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE, 

                                                              directory=validation_dir,

                                                              shuffle=False,

                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)

                                                              class_mode='binary')
sample_training_images, _ = next(train_data_gen) 
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

    plt.tight_layout()

    plt.show()
plotImages(sample_training_images[:5])  # Plot images 0-4
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)), #有32個卷積核,此次卷積將產生32張卷積圖片

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
EPOCHS = 100

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),

    epochs=EPOCHS,

    validation_data=val_data_gen,

    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))

)
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



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

plt.savefig('./foo.png')

plt.show()