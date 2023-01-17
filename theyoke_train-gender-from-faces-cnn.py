import tensorflow as tf



import matplotlib.pyplot as plt

import seaborn

import PIL.Image as Image

import numpy as np

import os



seaborn.set()
npz = np.load('/kaggle/input/faces-and-labeled-genders/fqjZGHsuVL3a.npz')

images, genders = npz['images'], npz['genders']



npz = np.load('/kaggle/input/get-imdb-wiki-image-dataset/imdb_dataset.npz')

images = np.append(images, npz['images'], axis=0)

genders = np.append(genders, npz['genders'], axis=0)



images_scaled = images[...,np.newaxis] / 255.0  # expand for channel axis and scale



del npz

images.shape, genders.shape
indices = np.random.choice(range(images.shape[0]), size=25, replace=False)



plt.figure(figsize=(10,10))

for i, j in enumerate(indices):

    gender = 'Male' if genders[j] else 'Female'

    

    plt.subplot(5,5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(images[j], cmap=plt.cm.binary_r)

    plt.xlabel(f'{gender}')

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(128, (6, 6), activation='relu', input_shape=(100, 100, 1)),

    tf.keras.layers.SpatialDropout2D(0.5),

    tf.keras.layers.AveragePooling2D((4, 4)),

    tf.keras.layers.Conv2D(128, (6, 6), activation='relu'),

    tf.keras.layers.SpatialDropout2D(0.5),

    tf.keras.layers.AveragePooling2D((4, 4)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
history = model.fit(images_scaled, genders, epochs=100, batch_size=200, validation_split=0.1, verbose=0)
model.evaluate(images_scaled, genders, batch_size=100, verbose=0)
plt.plot(history.epoch, history.history['accuracy'], '-', label='accuracy')

plt.plot(history.epoch, history.history['val_accuracy'], '--', label='val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.show()
plt.plot(history.epoch, history.history['loss'], '-', label='loss')

plt.plot(history.epoch, history.history['val_loss'], '--', label='val_loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
model.save('model.h5')