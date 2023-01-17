import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


BASE_DIR = '../input/100-bird-species/'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')
CATEGORIES = os.listdir(TRAIN_DIR)
train_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
)

validation_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)

test_data = ImageDataGenerator(
    rescale=1./255,
).flow_from_directory(
    VALIDATION_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
)
conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False

model = tf.keras.models.Sequential([
    conv_base,
    Flatten(),
    Dense(256, activation=tf.nn.relu),
    Dense(len(CATEGORIES), activation=tf.nn.softmax)
])

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.categorical_crossentropy,
    metrics=['accuracy'],
)
history = model.fit_generator(
    train_data,
    steps_per_epoch=280,
    validation_data=validation_data,
    validation_steps=14,
    epochs=27,
)
model.save('birds_weights.h5')
model.load_weights('birds_weights.h5')
_, accuracy = model.evaluate_generator(test_data)

print('Accuracy: ', round(accuracy * 100, 2), '%')

plt.plot(history.history['accuracy'], 'r-', label='Test Accuracy')
plt.plot(history.history['val_accuracy'], 'b-', label='Validation Accuracy')

plt.legend()
plt.grid()

plt.show()
