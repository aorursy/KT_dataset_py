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
from __future__ import absolute_import, division, print_function, unicode_literals



!pip install -q tensorflow==2.0.0-alpha0

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
!ls ../ -al
df=pd.read_csv("../input/train.csv")

#..나보다 하나 위의 디렉토리, 그 안의 input, 그 안의 train.csv file
df.sample(5)
id=df["category_id"]

id

#we want to see id of the animals
f=df["file_name"] 

f
#f is a panda data series

print(type(f))
#but we'd like to have a list

all_image_paths=['../input/train_images/'+fname for fname in f]

#specifies the directory '../input/train_images'

all_image_paths[:10]
#same thing for categroy name which will be used as labels

all_image_labels=[i for i in id]

all_image_labels[:10]
import random

import IPython.display as display
for n in range(3):

  image_path = random.choice(all_image_paths)

  display.display(display.Image(image_path))

  #print(caption_image(image_path))

  print()
img_path=all_image_paths[10]

img_path
img_raw = tf.io.read_file(image_path)

print(repr(img_raw)[:100]+"...")

#repr returns a printable representational string of a given object 

#and then with [:100] we are extracting until the 100th string 

#and ading "..."to be cool
img_tensor = tf.image.decode_image(img_raw)



print(img_tensor.shape)

print(img_tensor.dtype)
img_final=tf.image.resize(img_tensor,[192,192])

img_final=img_final/255.0

print(img_final.shape)

print(img_final.numpy().min())

print(img_final.numpy().max())
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
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(20):

    print(label.numpy())

#print(label_names[label.numpy()])-we have never defined label_names 
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))



# The tuples are unpacked into the positional arguments of the mapped function

def load_and_preprocess_from_path_label(path, label):

  return load_and_preprocess_image(path), label



image_label_ds = ds.map(load_and_preprocess_from_path_label)

image_label_ds
image_count= 1000
BATCH_SIZE = 32



# Setting a shuffle buffer size as large as the dataset ensures that the data is

# completely shuffled.

ds = image_label_ds.shuffle(buffer_size=image_count)

ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

# `prefetch` lets the dataset fetch batches, in the background while the model is training.

ds = ds.prefetch(buffer_size=AUTOTUNE)

ds
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)

mobile_net.trainable = False
def change_range(image,label):

  return 2*image-1, label



keras_ds = ds.map(change_range)
# The dataset may take a few seconds to start, as it fills its shuffle buffer.

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)

print(feature_map_batch.shape)
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
model.fit(ds,epochs=5, steps_per_epoch=3)