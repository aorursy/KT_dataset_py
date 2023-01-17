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
!pip install  tf-nightly
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

img_height = 250

img_width = 250

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
print(train_ds)
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
for imagt_batch, labelt_batch in val_ds.take(1):

  print(imagt_batch.shape)

  print(labelt_batch.shape)

  break
IMG_SIZE = 250 # All images will be resized to 160x160



def format_example(image, label):

    image = (image/127.5) - 1

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label
train = train_ds.map(format_example)

validation = val_ds.map(format_example)

validation
BATCH_SIZE = 32

SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE)

#validation_batches = validation.batch(BATCH_SIZE)

for imagee_batch, labeel_batch in val_ds.take(1):

   pass



imagee_batch.shape
for image_batch, label_batch in train_batches.take(1):

   pass



image_batch.shape
import tensorflow as tf

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D

from keras.layers import Conv1D,Conv2D,MaxPooling2D, MaxPooling1D, Reshape

from tensorflow.keras import datasets, layers, models 

import matplotlib.pyplot as plt

from tensorflow import keras
data_augmentation = keras.Sequential(

    [

        layers.experimental.preprocessing.RandomFlip("horizontal"),

        layers.experimental.preprocessing.RandomRotation(0.1),

        layers.experimental.preprocessing.RandomContrast(.2),

       

        layers.experimental.preprocessing.Normalization()





    ]

)
train_ds = train_ds.prefetch(buffer_size=30)

val_ds = val_ds.prefetch(buffer_size=30)
def make_model(input_shape, num_classes):

  



    inputs = keras.Input(shape=input_shape)

    # Image augmentation block

    x = data_augmentation(inputs)



    # Entry block

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 4, strides=2, padding="same")(x)

   # x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    x = layers.Conv2D(64,4, padding="same")(x)

    #x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    previous_block_activation = x  # Set aside residual



    for size in [ 512, 728]:#128, 256,

        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(size, 3, padding="same")(x)

       # x = layers.BatchNormalization()(x)



       # x = layers.Activation("relu")(x)

      #  x = layers.SeparableConv2D(size, 3, padding="same")(x)

     #   x = layers.BatchNormalization()(x)



        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)



        # Project residual

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(

            previous_block_activation

        )

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual



    x = layers.SeparableConv2D(1024, 4, padding="same")(x)

   # x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)



    x = layers.GlobalAveragePooling2D()(x)

    if num_classes == 2:

        activation = "sigmoid"

        units = 1

    else:

        activation = "softmax"

        units = num_classes



    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(5, activation="softmax")(x)

    return keras.Model(inputs, outputs)





model = make_model(input_shape=image_size + (3,), num_classes=5)

keras.utils.plot_model(model, show_shapes=True)
epochs = 50







callbacks = [

    tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=5,

    restore_best_weights=True

)

    

]

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(),

              metrics=['accuracy'])

model.fit(

    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,

)