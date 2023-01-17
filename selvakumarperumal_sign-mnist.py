import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
import os
import matplotlib.pyplot as plt
os.listdir('/kaggle/input/sign-language-mnist/')
def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')    
        imgs = []
        labels = []

        next(reader, None)
        
        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
    return images, labels
training_images, training_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')
testing_images, testing_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')

training_images.shape
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)
plt.imshow(training_images[30].reshape(28, 28),cmap='gray')
plt.show()
plt.imshow(training_images[50].reshape(28, 28),cmap='gray')
plt.show()

training_images.shape
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(
    rescale=1 / 255)
    
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')]
    )
train_gen = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=64)

val_gen = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=64)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit_generator(train_gen,
                              epochs=10,
                              validation_data=val_gen)