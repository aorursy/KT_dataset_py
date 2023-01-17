import random

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator



print("setup is ready")
train_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    "../input/hotdog-not/hot-dog-not-hot-dog/",

    target_size=(300, 300),

    batch_size=128,

    class_mode="binary"

)
print("final shape of images : ", train_generator.image_shape)
images, labels = next(train_generator)

plt.figure(figsize=(10,10))

for n in range(25):

    x = random.randrange(128)  

    ax = plt.subplot(5,5,n+1)

    plt.imshow(images[x])

    plt.title(labels[x])

    plt.axis('off')
# so zero means hotdog and one means not hotdog

class_names = ["hotdog", 'not hotdog']
images, labels = next(train_generator)

plt.figure(figsize=(10,15))

for n in range(25):

    x = random.randrange(128)  

    ax = plt.subplot(5,5,n+1)

    plt.imshow(images[x])

    plt.title(class_names[int(labels[x])])

    plt.axis('off')
sns.distplot(labels, kde=False, bins=20)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



print('MODEL CREATED')
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])



print("Model successfully compiled")
history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,  

      epochs=30,

      verbose=1)
plt.figure(figsize=(15,20))

for n in range(20):

    x = random.randrange(128)  

    ax = plt.subplot(4,5,n+1)

    plt.imshow(images[x])

    predicted = model.predict(images[x: x+1])[0][0]

    type = 0

    if predicted >= 0.5:

        type = 1

    plt.title(f"predicted  : {class_names[type]}")

    plt.axis('off')
train_datagen = ImageDataGenerator(rescale=1./255)



test_generator = train_datagen.flow_from_directory(

    "../input/yesornotest/Test/",

    target_size=(300, 300),

    batch_size=11,

    class_mode="binary"

)
images, labels = next(train_generator)

plt.figure(figsize=(15,20))

for n in range(10):  

    ax = plt.subplot(5,2,n+1)

    plt.imshow(images[n])

    predicted = model.predict(images[n: n+1])[0][0]

    type = 0

    if predicted >= 0.5:

        type = 1

    plt.title(f"predicted  : {class_names[type]}")

    plt.axis('off')