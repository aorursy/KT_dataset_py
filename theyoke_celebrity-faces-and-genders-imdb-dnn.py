import tensorflow as tf



import matplotlib.pyplot as plt

import seaborn

import PIL.Image as Image

import numpy as np



seaborn.set()
npz = np.load('/kaggle/input/get-imdb-wiki-image-dataset/imdb_dataset.npz')



images, ages, genders = npz['images'], npz['ages'], npz['genders']

images_scaled = images[...,np.newaxis] / 255.0  # expand for channel axis and scale



images.shape, ages.shape, genders.shape
indices = np.arange(images.shape[0])

np.random.shuffle(indices)



plt.figure(figsize=(10,10))

for i in range(36):

    gender = 'Male' if genders[indices[i]] else 'Female'

    age = ages[indices[i]]

    

    plt.subplot(6,6,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(images[indices[i]], cmap=plt.cm.binary_r)

    plt.xlabel(f'{gender} ({age})')

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(100, 100, 1)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.MaxPool2D((4, 4)),

    tf.keras.layers.Conv2D(96, (5, 5), activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.MaxPool2D((4, 4)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
history = model.fit(images_scaled, genders, epochs=100, batch_size=50, validation_split=0.2, verbose=2)
plt.plot(history.epoch, history.history['accuracy'], '-', label='accuracy')

plt.plot(history.epoch, history.history['val_accuracy'], '--', label='val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')

plt.show()
model.save('model.h5')