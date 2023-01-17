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
from __future__ import absolute_import, division, print_function

 

import tensorflow as tf

print(tf.VERSION)

  

tf.enable_eager_execution()

 

AUTOTUNE = tf.data.experimental.AUTOTUNE

 

import IPython.display as display

import matplotlib.pyplot as plt
#df = pd.read_csv('../input/iwildcam-2019-fgvc6/train.csv')

df = pd.read_csv('../input/train.csv')

  

f = df['file_name']

id = df['category_id']

 

all_image_paths = ['../input/train_images/' + fname for fname in f]

all_image_labels = [i for i in id]
print(all_image_paths[:10])
print(all_image_labels[:10])
img = open(all_image_paths[0], 'rb').read()

print(repr(img)[:100])
img = open(all_image_paths[0], 'rb').read() #byte string 

print(repr(img)[:100])

display.Image(img)

# works perfectly.





# another way to open the file

img2 = tf.io.read_file(all_image_paths[0])

print(repr(img2)[:100]+"...")

display.display(display.Image(img2.numpy()))

display.Image(img2.numpy())



# loading image files in two different ways
#import matplotlib.pyplot as plt

img = tf.io.read_file(all_image_paths[110735]) # smart tf.method, so that no need to spefify the type of each file 

a = tf.io.decode_jpeg(img) # to run the plt.imshow, we need to decode the above one so that the function can read 

plt.imshow(a) # file showing function 

print(a) # lets see how things were decoded
img3 = tf.io.encode_jpeg(a)
display.Image(img3.numpy()) # image encoding and decoding 
img = tf.io.read_file(all_image_paths[40252])

a = tf.io.decode_jpeg(img)

print(a.shape)

temp = tf.image.resize_images(a, [28,28])

temp = tf.dtypes.cast(temp, tf.uint8) #cast: transform one data type into another especially in "dtypes.cast"

# unit8: unsinged integer, smallest ones all zeros, biggest ones are all ones 

b = tf.io.encode_jpeg(temp)

display.Image(b.numpy())
def _bytes_feature(value):

  """Returns a bytes_list from a string / byte."""

  if isinstance(value, type(tf.constant(0))):

    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
img5 = tf.io.read_file(all_image_paths[40252])

#_bytes_feature(img5) #smaller
paths_labels = dict(zip(all_image_paths[0:10], all_image_labels[0:10]))
def image_example(image_string, label):

    feature = {

      'label': _int64_feature(label),

      'image_raw': _bytes_feature(image_string),

  }

    return tf.train.Example(features=tf.train.Features(feature=feature))
record_file = 'images.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:

    for filename, label in paths_labels.items():

        image_string = open(filename, 'rb').read()  

        tf_example = image_example(image_string, label)

        writer.write(tf_example.SerializeToString()) #evrying should be in a one sequence
!ls -al
#!ls ../input/train_images/ -al
paths_labels2 = dict(zip(all_image_paths[0:1000], all_image_labels[0:10]))

record_file = 'images.tfrecords2'

with tf.io.TFRecordWriter(record_file) as writer:

    for filename, label in paths_labels2.items():

        #image_string = open(filename, 'rb').read() 

        image_string = tf.io.read_file(filename)

        tf_example = image_example(image_string, label)

        writer.write(tf_example.SerializeToString())
#!ls -al
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
def parse(x):

    feature = {'image_raw':tf.io.FixedLenFeature([],tf.string),

             'label':tf.io.FixedLenFeature([],tf.int64)}

    return tf.io.parse_single_example(x,feature)
ds = raw_image_dataset.map(parse)
# for i in ds.take(1):

#     print(i['image_raw'].numpy())
for i in ds:

    print(i['label'].numpy())
ds = ds.shuffle(buffer_size=10) # take the specified number of buffer_size, and spits out randomly withing the batch

#ds = ds.shuffle(buffer_size=1) #first one comes and that same thing goes out, so bascially same order 



for i in ds:

    print(i['label'].numpy())
ds = ds.repeat(2) # if "ds.repeat(), this will repeat forever"



for i in ds:

    print(i['label'].numpy())
#ds = ds.batch(1)

ds = ds.batch(2) # take the specified number of units at once

for i in ds:

    print(i['label'].numpy())
ds = ds.batch(2) 

for i in ds:

    print(i['label'].numpy())