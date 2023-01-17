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

import IPython.display as display



tf.enable_eager_execution()

print(tf.VERSION)



AUTOTUNE = tf.data.experimental.AUTOTUNE
df = pd.read_csv('../input/train.csv')



f = df['file_name']

id = df['category_id']



all_image_paths = ['../input/train_images/' + fname for fname in f]

all_image_labels = [i for i in id]



paths_labels = dict(zip(all_image_paths[0:10], all_image_labels[0:10]))
#display.display(display.Image(all_image_paths[23]))
def preprocess_image(image):

  image = tf.image.decode_jpeg(image, channels=3)

  image = tf.image.resize_images(image, [192, 192])

  image /= 255.0  # normalize to [0,1] range



  return image



def load_and_preprocess_image(path):

  image = tf.read_file(path)

  return preprocess_image(image)
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))



# The tuples are unpacked into the positional arguments of the mapped function 

def load_and_preprocess_from_path_label(path, label):

  return load_and_preprocess_image(path), label



image_label_ds = ds.map(load_and_preprocess_from_path_label)

image_label_ds
BATCH_SIZE = 32



# Setting a shuffle buffer size as large as the dataset ensures that the data is

# completely shuffled.

ds = image_label_ds.shuffle(buffer_size=1)

ds = ds.repeat()

ds = ds.batch(BATCH_SIZE)

# `prefetch` lets the dataset fetch batches, in the background while the model is training.

ds = ds.prefetch(buffer_size=AUTOTUNE)

ds
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)

mobile_net.trainable=False
def change_range(image,label):

  return 2*image-1, label



keras_ds = ds.map(change_range)
model = tf.keras.Sequential([

  mobile_net,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(23)])
model.compile(optimizer=tf.train.AdamOptimizer(), 

              loss=tf.keras.losses.sparse_categorical_crossentropy,

              metrics=["accuracy"])
model.summary()
steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

steps_per_epoch
# It takes more than 2 hours...

#model.fit(keras_ds, epochs=1, steps_per_epoch=6135)
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
#_bytes_feature(b'3')
def image_example(image_string, label):



  feature = {

      'label': _int64_feature(label),

      'image_raw': _bytes_feature(image_string),

  }



  return tf.train.Example(features=tf.train.Features(feature=feature))
%%time



paths_labels = dict(zip(all_image_paths[0:10], all_image_labels[0:10]))

record_file = 'images.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:

  for filename, label in paths_labels.items():

#    image_string = open(filename, 'rb').read() 

    image_string = tf.io.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string)

    image_resized = tf.image.resize_images(image_decoded, (28,28))

#    image_resized = tf.image.resize_images(image_decoded, (192,192))

#    image_bytes = image_resized.numpy().tobytes()

    image_casted = tf.dtypes.cast(image_resized, tf.uint8)

    image_bytes = tf.image.encode_jpeg(image_casted)

    tf_example = image_example(image_bytes, label)

    writer.write(tf_example.SerializeToString())
!ls -al
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')



#def _parse_(serialized_example):

#    feature = {'image_raw':tf.io.FixedLenFeature([],tf.string),

#               'label':tf.io.FixedLenFeature([],tf.int64)}

#    example = tf.io.parse_single_example(serialized_example,feature)

#    image = tf.io.decode_raw(example['image_raw'],tf.int64) #remember to parse in int64. float will raise error

#    label = tf.cast(example['label'],tf.int32)

#    return (dict({'image':image}),label)

#    return image, label





def parse(x):

  feature = {'image_raw':tf.io.FixedLenFeature([],tf.string),

             'label':tf.io.FixedLenFeature([],tf.int64)}

  return tf.io.parse_single_example(x,feature)
%%time



ds = raw_image_dataset.map(parse)
#model.fit (ds, epochs=1, steps_per_epoch=10)
def _parse_function(proto):

    # define your tfrecord again. Remember that you saved your image as a string.

    keys_to_features = {'image_raw': tf.FixedLenFeature([], tf.string),

                        "label": tf.FixedLenFeature([], tf.int64)}

    

    # Load one example

    parsed_features = tf.parse_single_example(proto, keys_to_features)

    

    # Turn your saved image string into an array

    parsed_features['image_raw'] = tf.image.decode_jpeg(parsed_features['image_raw'])

    

    return parsed_features['image_raw'], parsed_features["label"]



  

def create_dataset(filepath):

    

    # This works with arrays as well

    dataset = tf.data.TFRecordDataset(filepath)

    

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here

    dataset = dataset.map(_parse_function, num_parallel_calls=1)

    

    # This dataset will go on forever

    dataset = dataset.repeat()

    

    # Set the number of datapoints you want to load and shuffle 

    #dataset = dataset.shuffle(1)

    

    # Set the batchsize

    dataset = dataset.batch(100)

    

    # Create an iterator

    iterator = dataset.make_one_shot_iterator()

    

    # Create your tf representation of the iterator

    image, label = iterator.get_next()

    # Bring your picture back in shape

    image = tf.reshape(image, [-1,28,28,3])

    image = tf.cast(image, dtype=tf.uint8)

    image = image.numpy()/255.

    

    # Create a one hot array for your labels

#    label = tf.one_hot(label, 23)

    

    return image, label
image, label = create_dataset('images.tfrecords')



print(image.shape)

image=image.reshape(-1,28,28,3)

#import tensorflow as tf

#mnist = tf.keras.datasets.mnist



#(x_train, y_train),(x_test, y_test) = mnist.load_data()

#x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=image

y_train=label



model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape= (28, 28, 3)),

  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(23, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=5, steps_per_epoch=100) #validation_split=0.1)

#model.evaluate(x_test, y_test)

ds = ds.shuffle(buffer_size=2)



for i in ds:

  print(i['label'].numpy())

  print(i)

#  display.display(display.Image(i['image_raw'].numpy()))
ds = ds.repeat(2)



for i in ds:

  print(i['label'].numpy())

#  display.display(display.Image(i['image_raw'].numpy()))
ds = ds.batch(1)

for i in ds:

  print(i['label'].numpy())

#  display.display(display.Image(i['image_raw'].numpy()))
# iterator = tf.compat.v1.data.make_one_shot_iterator(ds)