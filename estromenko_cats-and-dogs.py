import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import os
import zipfile


CATEGORIES = ['Cat', 'Dog']

DATA_DIR = '/kaggle/working/train'
BASE_DIR = '/kaggle/working/cats_and_dogs'

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

TRAIN_CATS_DIR = os.path.join(TRAIN_DIR, 'cats')
TRAIN_DOGS_DIR = os.path.join(TRAIN_DIR, 'dogs')

TEST_CATS_DIR = os.path.join(TEST_DIR, 'cats')
TEST_DOGS_DIR = os.path.join(TEST_DIR, 'dogs')

VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, 'cats')
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, 'dogs')
START_DATA_DIR = '/kaggle/input/dogs-vs-cats-redux-kernels-edition'

with zipfile.ZipFile(os.path.join(START_DATA_DIR, 'train.zip'), 'r') as z:
    z.extractall('/kaggle/working')

os.mkdir(BASE_DIR)
os.mkdir(TRAIN_DIR)
os.mkdir(TEST_DIR)
os.mkdir(VALIDATION_DIR)

os.mkdir(TRAIN_CATS_DIR)
os.mkdir(TRAIN_DOGS_DIR)

os.mkdir(TEST_CATS_DIR)
os.mkdir(TEST_DOGS_DIR)

os.mkdir(VALIDATION_CATS_DIR)
os.mkdir(VALIDATION_DOGS_DIR)


cats_images = ['cat.{}.jpg'.format(i) for i in range(5000)]
for image in cats_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(TRAIN_CATS_DIR, image)
    shutil.copyfile(fr, to)

cats_images = ['cat.{}.jpg'.format(i) for i in range(5000, 10000)]
for image in cats_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(VALIDATION_CATS_DIR, image)
    shutil.copyfile(fr, to)
    
cats_images = ['cat.{}.jpg'.format(i) for i in range(10000, 12500)]
for image in cats_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(TEST_CATS_DIR, image)
    shutil.copyfile(fr, to)
    
    
    
    
dogs_images = ['dog.{}.jpg'.format(i) for i in range(5000)]
for image in dogs_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(TRAIN_DOGS_DIR, image)
    shutil.copyfile(fr, to)

dogs_images = ['dog.{}.jpg'.format(i) for i in range(5000, 10000)]
for image in dogs_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(VALIDATION_DOGS_DIR, image)
    shutil.copyfile(fr, to)
    
dogs_images = ['dog.{}.jpg'.format(i) for i in range(10000, 12500)]
for image in dogs_images:
    fr = os.path.join(DATA_DIR, image)
    to = os.path.join(TEST_DOGS_DIR, image)
    shutil.copyfile(fr, to)
    
train_generator = ImageDataGenerator(
    rescale = 1. / 255,
).flow_from_directory(
    TRAIN_DIR,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary',
)

validation_generator = ImageDataGenerator(
    rescale = 1. / 255,
).flow_from_directory(
    TEST_DIR,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = 'binary',
)
conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150, 150, 3))
conv_base.trainable = False
conv_base.summary()

model = tf.keras.models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss = losses.binary_crossentropy,
    optimizer = optimizers.RMSprop(lr=0.001),
    metrics = ['accuracy'],
)

model.summary()
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    validation_data = validation_generator,
    validation_steps = 50,
    epochs = 30,
)

model.save('cats_and_dogs.h5')
model.load_weights('cats_and_dogs.h5')
_, accuracy = model.evaluate_generator(validation_generator)
print('Accuracy: ', str(round(accuracy * 100, 2)), '%')
    
for img in validation_generator:
    image = img[0]
    label = img[1]
    break    

index = 0
model_prediction = CATEGORIES[int(round(model.predict(image)[index][0]))]
plt.imshow(image[index])
print(model_prediction)
