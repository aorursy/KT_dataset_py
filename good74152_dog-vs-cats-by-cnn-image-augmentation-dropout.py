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

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'



zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
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
BATCH_SIZE = 100

IMG_SHAPE  = 150 # Our training data consists of images with width of 150 pixels and height of 150 pixels
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

    plt.tight_layout()

    plt.show()
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)



train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, 

                                               directory=train_dir, 

                                               shuffle=True, 

                                               target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)



train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, 

                                               directory=train_dir, 

                                               shuffle=True, 

                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)



train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, 

                                               directory=train_dir, 

                                               shuffle=True, 

                                               target_size=(IMG_SHAPE, IMG_SHAPE))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen_train = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, 

                                                     directory=train_dir, 

                                                     shuffle=True, 

                                                     target_size=(IMG_SHAPE,IMG_SHAPE),

                                                     class_mode='binary')
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen_val = ImageDataGenerator(rescale=1./255)



val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE, 

                                                 directory=validation_dir, 

                                                 target_size=(IMG_SHAPE, IMG_SHAPE),

                                                 class_mode='binary')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Dropout(0.5),#隨機關閉一半的神經元, 不讓權重過大的神經元過度影響整個神經網路

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
epochs=100

history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))

)
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



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