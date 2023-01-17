# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pathlib  # path management

print('Tensorflow version: {}'.format(tf.__version__))
data_dir = '/kaggle/input/lake-plane-binaryclass/2_class'
data_root = pathlib.Path(data_dir)
data_root
# list all dir

for item in data_root.iterdir():

    print(item)
# list all files

all_image_paths = list(data_root.glob('*/*'))
image_count = len(all_image_paths)

image_count
all_image_paths[:3]
all_image_paths[-3:]
import random

# change to real paths

all_image_paths = [str(path) for path in all_image_paths] 



# shuffle

random.shuffle(all_image_paths)



image_count = len(all_image_paths)

image_count
all_image_paths[:5]
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

label_names
label_to_index = dict((name, index) for index,name in enumerate(label_names))

label_to_index
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
all_image_labels[:5]
import IPython.display as display
def caption_image(label):

    return {0: 'airplane', 1: 'lake'}.get(label)
for n in range(3):

    image_index = random.choice(range(len(all_image_paths)))

    display.display(display.Image(all_image_paths[image_index]))

    print(caption_image(all_image_labels[image_index]))

    print()
# image path

img_path = all_image_paths[0]

img_path
# tensorflow read image data

img_raw = tf.io.read_file(img_path)

print(repr(img_raw)[:100]+"...")
# decode image

img_tensor = tf.image.decode_image(img_raw)



print(img_tensor.shape)

print(img_tensor.dtype)
# change data type from uint8 to float32

img_tensor = tf.cast(img_tensor, tf.float32)

# Normalize to [0,1] range

img_final = img_tensor/255.0

print(img_final.shape)

print(img_final.numpy().min())  # img_final.numpy(): change to ndarray

print(img_final.numpy().max())
def load_and_preprocess_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [256, 256])

    image = tf.cast(image, tf.float32)

    image = image/255.0  # normalize to [0,1] range

    return image
import matplotlib.pyplot as plt



n=101

image_path = all_image_paths[n]

label = all_image_labels[n]



plt.imshow(load_and_preprocess_image(image_path))

plt.grid(False)

plt.xlabel(caption_image(label))
# all image paths dataset

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# all image dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
image_ds
# all label dataset

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int32))
label_ds
# show the first 10 label

for label in label_ds.take(10):

    print(label_names[label.numpy()])
# zip to one dataset 

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

image_label_ds
test_count = int(image_count*0.2)

train_count = image_count - test_count

train_count,test_count
train_data = image_label_ds.skip(test_count)

test_data = image_label_ds.take(test_count)
# 每次训练的图片数

BATCH_SIZE = 32
train_data = train_data.apply(

  tf.data.experimental.shuffle_and_repeat(buffer_size=train_count))

  #tf.data.dataset.repeat.shuffle(buffer_size=train_count))

train_data=train_data.batch(BATCH_SIZE)

train_data=train_data.prefetch(buffer_size = AUTOTUNE)

train_data
test_data = test_data.batch(BATCH_SIZE)
test_data
# Build model

model = tf.keras.Sequential()   #顺序模型

model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(1024, activation='relu'))

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc']

)
steps_per_epoch = train_count//BATCH_SIZE

validation_steps = test_count//BATCH_SIZE
history = model.fit(train_data, epochs=20, steps_per_epoch=steps_per_epoch, validation_data=test_data, validation_steps=validation_steps)
history.history.keys()
plt.plot(history.epoch, history.history.get('acc'), label='acc')

plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')

plt.legend()
plt.plot(history.epoch, history.history.get('loss'), label='loss')

plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')

plt.legend()
model.save('lake-plane.h5')