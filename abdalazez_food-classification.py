# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -q tf-nightly
import numpy as np

import os

import PIL

import PIL.Image

import tensorflow as tf

print(tf.__version__)
! cp -R ../input/recipes ./myrecipes
import pathlib

data_dir = './myrecipes'

data_dir = pathlib.Path(data_dir)

img_list = list(data_dir.glob('*/*.*'))

image_count = len(img_list)

print(image_count)
img_list[0]
import os.path



img_ext = dict()

for img in img_list:

    extension = os.path.splitext(str(img))[1]

    img_ext[extension] = img_ext.get(extension, 0) + 1

img_ext
import os



num_skipped = 0



for folder_name in ['briyani', 'burger', 'dosa', 'idly', 'pizza']:

    folder_path = os.path.join("./myrecipes/", folder_name)

    for fname in os.listdir(folder_path):

        fpath = os.path.join(folder_path, fname)

        try:

            fobj = open(fpath, "rb")

            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)

        finally:

            fobj.close()



        if not is_jfif:

            num_skipped += 1

            # Delete corrupted image

            os.remove(fpath)



print("Deleted %d images" % num_skipped)
import os.path

data_dir = './myrecipes'

data_dir = pathlib.Path(data_dir)

img_list = list(data_dir.glob('*/*.*'))

image_count = len(img_list)

print(image_count)

img_ext = dict()

for img in img_list:

    extension = os.path.splitext(str(img))[1]

    img_ext[extension] = img_ext.get(extension, 0) + 1

print(img_ext)
batch_size = 32

epochs = 50

img_height = 250

img_width = 250

image_size = img_height, img_width
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



#train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

#validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    './myrecipes/',

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=image_size,

    batch_size=batch_size,

)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    './myrecipes/',

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=image_size,

    batch_size=batch_size,

)
class_names = train_ds.class_names

class_names
import matplotlib.pyplot as plt



plt.figure(figsize=(16, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[int(labels[i])])

        plt.axis("off")
for image_batch, labels_batch in train_ds:

  print(image_batch.shape)

  print(labels_batch.shape)

  break
print(os.listdir('./myrecipes/'))
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
data_augmentation = keras.Sequential([

        layers.experimental.preprocessing.RandomFlip("horizontal"),

        layers.experimental.preprocessing.RandomRotation(0.1),

])
augmented_train_ds = train_ds.map(

  lambda x, y: (data_augmentation(x, training=True), y))

augmented_val_ds = val_ds.map(

  lambda x, y: (data_augmentation(x, training=True), y))



#augmented_train_ds = train_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

#augmented_val_ds = val_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

# Our vectorized labels

#y_train = np.asarray(train_ds).astype('float32').reshape((-1,1))

#y_test = np.asarray(val_ds).astype('float32').reshape((-1,1))
augmented_train_ds
augmented_val_ds
model = Sequential([

    Conv2D(16, 5, padding='same', activation='relu', input_shape=(img_height, img_width ,3)),

    MaxPooling2D(),

    Dropout(0.25),

    Conv2D(32, 5, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.25),

    Conv2D(64, 5, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(5)

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
history = model.fit(

    augmented_train_ds,

    steps_per_epoch=5,

    epochs=epochs,

    validation_data=augmented_val_ds,

    validation_steps=5

)