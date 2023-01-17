import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
data = tf.keras.datasets.cifar10
((train_images, train_labels), (test_images, test_labels)) = data.load_data()
print('train_images :', train_images.shape)

print('train_labels :', train_labels.shape)

print('test_images :', test_images.shape)

print('test_images :', test_labels.shape)
fig = plt.figure(figsize=(8, 8))

rows, cols = 5, 4



for i in range(1, rows*cols+1):

    fig.add_subplot(rows, cols, i)

    plt.imshow(train_images[i])

    plt.axis('off')

plt.show()
train_images_flat = train_images.reshape((-1, 32*32*3))

print(train_images_flat.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model = Sequential([

    Dense(500, input_shape=(3072,), activation='relu'),

    Dense(100, activation='relu'),

    Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_images_flat / 255,

                    y=train_labels,

                    validation_split=0.1, # 50000 -> 45000 / 5000

                    epochs=30,

                    batch_size=32,

                    verbose=False)
def plot_history(history):

    train_loss = history.history['loss']

    valid_loss = history.history['val_loss']

    

    train_acc = history.history['accuracy']

    valid_acc = history.history['val_accuracy']

    

    plt.figure(figsize=(8, 8))

    

    plt.subplot(2, 1, 1)

    plt.plot(train_loss, label='Train Loss')

    plt.plot(valid_loss, label='Validation Loss')

    plt.legend()

    plt.title('Train Loss vs Validation Loss')

    

    plt.subplot(2, 1, 2)

    plt.plot(train_acc, label='Train Accuracy')

    plt.plot(valid_acc, label='Validation Accuracy')

    plt.legend()

    plt.title('Train Accuracy vs Validation Accuracy')

    

    plt.show()
plot_history(history)
from tensorflow.keras.layers import BatchNormalization, Dropout, ReLU
model = Sequential([

    Dense(500, input_shape=(3072,)),

    BatchNormalization(),

    Dropout(0.5),

    ReLU(),

    

    Dense(100),

    BatchNormalization(),

    Dropout(0.5),

    ReLU(),

    

    Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_images_flat / 255,

                    y=train_labels,

                    validation_split=0.1, # 50000 -> 45000 / 5000

                    epochs=30,

                    batch_size=32,

                    verbose=False)
plot_history(history)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    MaxPooling2D(2, 2),

    

    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(2, 2),

    

    Conv2D(128, (3, 3), activation='relu'),

    

    Flatten(),

    

    Dense(500),

    BatchNormalization(),

    Dropout(0.5),

    ReLU(),

    

    Dense(100),

    BatchNormalization(),

    Dropout(0.5),

    ReLU(),

    

    Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_images / 255,

                    y=train_labels,

                    validation_split=0.1, # 50000 -> 45000 / 5000

                    epochs=30,

                    batch_size=32,

                    verbose=False)
plot_history(history)