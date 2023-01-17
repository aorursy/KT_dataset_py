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
import numpy as np



from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input, decode_predictions



from tensorflow.python.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator



batch_size=163





#Create training data generator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        width_shift_range=0.1,

        height_shift_range=0.1) 
train_generator = train_datagen.flow_from_directory(

        './myrecipes/', #directory that contains training data

        target_size=(150, 150), #what size image we want

        batch_size=batch_size, #how many files to read in at a time

        class_mode="categorical")
from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(250,250,3), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2)))





model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(4,4), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(filters=128, kernel_size=(4,4), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2))) 



model.add(Conv2D(filters=128, kernel_size=(4,4), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2))) 



tf.keras.layers.Dropout(0.2),



model.add(Flatten())





model.add(Dense(256, activation='relu'))



model.add(Dense(len(class_names), activation='softmax'))

model.summary()

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
train_ds = train_ds.prefetch(buffer_size=32)

val_ds = val_ds.prefetch(buffer_size=32)
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping 
epochs = 100



model_cp = keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)





model.compile(

    optimizer=keras.optimizers.Adam(1e-3),

    loss="binary_crossentropy",

    metrics=["accuracy"],

)

history = model.fit(

    train_ds, epochs=epochs, callbacks=[model_cp, earlystop], validation_data=val_ds

)