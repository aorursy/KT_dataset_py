# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_width, img_height = 28,28

#defining the data directories
train_data_dir= '../input/horses-or-humans-dataset/horse-or-human/train'
validation_data_dir= '../input/horses-or-humans-dataset/horse-or-human/validation'
n_training_sample= 400
n_validation_sample= 100
epochs=20
batch_size=10
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(128, activation=tf.nn.relu), 
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)])
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=n_training_sample // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=n_validation_sample // batch_size)
pred= image.load_img('../input/horses-or-humans-dataset/horse-or-human/validation/horses/horse1-000.png', target_size=(28,28))
pred=image.img_to_array(pred)
pred= np.expand_dims(pred, axis=0)
result= model.predict(pred)
print(result)

if result[0][0]==1:
    answer='Horse'
else:
    answer='Human'
print("The image is a ",answer)
pred= image.load_img('../input/horses-or-humans-dataset/horse-or-human/validation/humans/valhuman01-09.png', target_size=(28,28))
pred=image.img_to_array(pred)
pred= np.expand_dims(pred, axis=0)
result= model.predict(pred)
print(result)
if result[0][0]==1:
    answer='Horse'
else:
    answer='Human'
print("The image is a ",answer)
