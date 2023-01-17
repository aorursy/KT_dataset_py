# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
df.sample(5)
df.columns.tolist()
from __future__ import absolute_import, division, print_function, unicode_literals



!pip install -q tensorflow==2.0.0-alpha0

import tensorflow as tf



'''

from

https://www.tensorflow.org/alpha/tutorials/load_data/images



'''
AUTOTUNE = tf.data.experimental.AUTOTUNE
id = df['category_id']

id.unique
id[:10] #contains category numbers
f = df['file_name']

f # like the 'all_image_paths'
all_image_paths = ['../input/train_images/'+fname for fname in f] 

#to specify the location of each file
for fname in f[:10]:

    print(fname) 
all_image_paths[:5]
import IPython.display as display
import random
for n in range(3):

    image_path = random.choice(all_image_paths)

    display.display(display.Image(image_path))

    print()
all_image_labels = [i for i in id]
all_image_labels[:5]
img_path = all_image_paths[0]

img_path
img_raw = tf.io.read_file(img_path)

print(repr(img_raw)[:100]+"...")
img_tensor = tf.image.decode_image(img_raw)



print(img_tensor.shape)

print(img_tensor.dtype)
# resize

img_final = tf.image.resize(img_tensor, [192, 192])

img_final = img_final/255.0

print(img_final.shape)

print(img_final.numpy().min())

print(img_final.numpy().max())
# wrap everything in a function

def preprocess_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [192, 192])

    image /= 255.0  # normalize to [0,1] range



    return image
def load_and_preprocess_image(path):

    image = tf.io.read_file(path)

    return preprocess_image(image)
import matplotlib.pyplot as plt



image_path = all_image_paths[0]

label = all_image_labels[0]



plt.imshow(load_and_preprocess_image(img_path))

plt.grid(False)

#plt.xlabel(caption_image(img_path))

#plt.title(label_names[label].title())

print()
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
import matplotlib.pyplot as plt



plt.figure(figsize=(8,8))

for n,image in enumerate(image_ds.take(4)):

    plt.subplot(2,2,n+1)

    plt.imshow(image)

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

  #plt.xlabel(caption_image(all_image_paths[n]))
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):

    print(label.numpy())
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))



# The tuples are unpacked into the positional arguments of the mapped function

def load_and_preprocess_from_path_label(path, label):

    return load_and_preprocess_image(path), label



image_label_ds = ds.map(load_and_preprocess_from_path_label)

image_label_ds
BATCH_SIZE = 32

#image_count = len(all_image_paths)

image_count = 100





# Setting a shuffle buffer size as large as the dataset ensures that the data is

# completely shuffled.

ds = image_label_ds.shuffle(buffer_size=image_count)

ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

# `prefetch` lets the dataset fetch batches, in the background while the model is training.

ds = ds.prefetch(buffer_size=AUTOTUNE)

ds
ds = image_label_ds.apply(

  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))

ds = ds.batch(BATCH_SIZE)

ds = ds.prefetch(buffer_size=AUTOTUNE)

ds
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)

mobile_net.trainable=False
#!help(keras_applications.mobilenet_v2.preprocess_input)
def change_range(image,label):

    return 2*image-1, label



keras_ds = ds.map(change_range)
# The dataset may take a few seconds to start, as it fills its shuffle buffer.

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)

print(feature_map_batch.shape)
class_num = id.unique()

type(class_num)

c = class_num.reshape(1,-1)

c
label.shape
model = tf.keras.Sequential([

  mobile_net,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(22)])
logit_batch = model(image_batch).numpy()



print("min logit:", logit_batch.min())

print("max logit:", logit_batch.max())

print()



print("Shape:", logit_batch.shape)
model.compile(optimizer=tf.keras.optimizers.Adam(),

              loss='sparse_categorical_crossentropy',

              metrics=["accuracy"])
len(model.trainable_variables)
model.summary()
steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

steps_per_epoch
model.fit(ds, epochs=1, steps_per_epoch=3)