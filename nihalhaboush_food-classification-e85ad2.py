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

img_height = 160

img_width = 160

image_size = img_height, img_width
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
import tensorflow_datasets as tfds



dataset, info =  tfds.load('mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']



num_train_examples= info.splits['train'].num_examples
def convert(image, label):

  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

  return image, label



def augment(image,label):

  image,label = convert(image, label)

  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]

  image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding

  image = tf.image.random_crop(image, size=[28, 28, 1]) # Random crop back to 28x28

  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness



  return image,label
AUTOTUNE = tf.data.experimental.AUTOTUNE

NUM_EXAMPLES = 2048

augmented_train_batches = (

    train_dataset

    # Only train on a subset, so you can quickly see the effect.

    .take(NUM_EXAMPLES)

    .cache()

    .shuffle(num_train_examples//4)

    # The augmentation is added here.

    .map(augment, num_parallel_calls=AUTOTUNE)

    .batch(batch_size)

    .prefetch(AUTOTUNE)

)

non_augmented_train_batches = (

    train_dataset

    # Only train on a subset, so you can quickly see the effect.

    .take(NUM_EXAMPLES)

    .cache()

    .shuffle(num_train_examples//4)

    # No augmentation.

    .map(convert, num_parallel_calls=AUTOTUNE)

    .batch(batch_size)

    .prefetch(AUTOTUNE)

) 
validation_batches = (

    test_dataset

    .map(convert, num_parallel_calls=AUTOTUNE)

    .batch(2*batch_size)

)
def make_model():

  model = tf.keras.Sequential([

      layers.Flatten(input_shape=(28, 28, 1)),

      layers.Dense(4096, activation='relu'),

      layers.Dense(4096, activation='relu'),

      layers.Dense(10)

  ])

  model.compile(optimizer = 'adam',

                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),

                metrics=['accuracy'])

  return model
from tensorflow.keras import layers





model_without_aug = make_model()



no_aug_history = model_without_aug.fit(non_augmented_train_batches, epochs=100, validation_data=validation_batches)
model_with_aug = make_model()



aug_history = model_with_aug.fit(augmented_train_batches, epochs=100, validation_data=validation_batches)