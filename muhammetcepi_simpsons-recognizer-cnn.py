from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt
PATH="/kaggle/input/simpsons_new/"
train_dir = os.path.join(PATH, 'train_set')

validation_dir = os.path.join(PATH, 'test_set')
train_homer_simpson_dir = os.path.join(train_dir, 'homer_simpson')  # directory with our training homer simpson pictures

train_ned_flanders_dir = os.path.join(train_dir, 'ned_flanders')  # directory with our training ned flanders pictures

train_bart_simpson_dir = os.path.join(train_dir, 'bart_simpson')  # directory with our training bart simpson pictures

train_moe_szyslak_dir = os.path.join(train_dir, 'moe_szyslak')  # directory with our training moe szyslak pictures

validation_homer_simpson_dir = os.path.join(validation_dir, 'homer_simpson')  # directory with our validation homer simpson pictures

validation_ned_flanders_dir = os.path.join(validation_dir, 'ned_flanders')  # directory with our validation ned flanders pictures

validation_bart_simpson_dir = os.path.join(validation_dir, 'bart_simpson')  # directory with our validation bart simpson pictures

validation_moe_szyslak_dir = os.path.join(validation_dir, 'moe_szyslak')  # directory with our validation homer simpson pictures
num_homer_simpson_tr = len(os.listdir(train_homer_simpson_dir))

num_ned_flanders_tr = len(os.listdir(train_ned_flanders_dir))

num_bart_simpson_tr = len(os.listdir(train_bart_simpson_dir))

num_moe_szyslak_tr = len(os.listdir(train_moe_szyslak_dir))



num_homer_simpson_val = len(os.listdir(validation_homer_simpson_dir))

num_ned_flanders_val = len(os.listdir(validation_ned_flanders_dir))

num_bart_simpson_val = len(os.listdir(validation_bart_simpson_dir))

num_moe_szyslak_val = len(os.listdir(validation_moe_szyslak_dir))



total_train = num_homer_simpson_tr + num_ned_flanders_tr+num_bart_simpson_tr+num_moe_szyslak_tr

total_val = num_homer_simpson_val + num_ned_flanders_val+num_bart_simpson_val+num_moe_szyslak_val
print('total training homer_simpson images:', num_homer_simpson_tr)

print('total training ned_flanders images:', num_ned_flanders_tr)

print('total training bart_simpson images:', num_bart_simpson_tr)

print('total training moe_szyslak images:', num_moe_szyslak_tr)



print('total validation homer_simpson images:', num_homer_simpson_val)

print('total validation ned_flanders images:', num_ned_flanders_val)

print('total validation bart_simpson images:', num_bart_simpson_val)

print('total validation moe_szyslak images:', num_moe_szyslak_val)

print("--")

print("Total training images:", total_train)

print("Total validation images:", total_val)
batch_size = 50

epochs = 15

IMG_HEIGHT = 150

IMG_WIDTH = 150
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='categorical')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory=validation_dir,

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='categorical')
sample_training_images, _ = next(train_data_gen)
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
plotImages(sample_training_images[:5])
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(32, activation='relu'),

    Dense(16, activation='relu'),

    Dense(4, activation='softmax')

])
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



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
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_HEIGHT, IMG_WIDTH))



augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=True,

                    zoom_range=0.5

                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                     class_mode='categorical')
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                 directory=validation_dir,

                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                 class_mode='categorical')
model_new = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', 

           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Dropout(0.2),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(4, activation='softmax')

])
model_new.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model_new.summary()
history = model_new.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=50,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(50)



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