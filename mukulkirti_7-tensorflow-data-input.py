import pathlib

import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import tensorflow.compat.v1 as tf

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

dataset
for elem in dataset:

  print(elem.numpy())
#Or by explicitly creating a Python iterator using iter and consuming its elements using next:



it = iter(dataset)



print(next(it).numpy())
print(dataset.reduce(0, lambda state, value: state + value).numpy())
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

dataset1.element_spec
dataset2 = tf.data.Dataset.from_tensor_slices(

   (tf.random.uniform([4]),

    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))



dataset2.element_spec
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3.element_spec
# Dataset containing a sparse tensor.

dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

dataset4.element_spec
dataset1 = tf.data.Dataset.from_tensor_slices(

    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

dataset1
for z in dataset1:

  print(z.numpy())
dataset2 = tf.data.Dataset.from_tensor_slices(

   (tf.random.uniform([4]),

    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2
for a, (b,c) in dataset3:

  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))
train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train

images = images/255



dataset = tf.data.Dataset.from_tensor_slices((images, labels))

dataset
def count(stop):

  i = 0

  while i<stop:

    yield i

    i += 1



for n in count(5):

  print(n)
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )
for count_batch in ds_counter.repeat().batch(10).take(10):

  print(count_batch.numpy())
def gen_series():

  i = 0

  while True:

    size = np.random.randint(0, 10)

    yield i, np.random.normal(size=(size,))

    i += 1



for i, series in gen_series():

  print(i, ":", str(series))

  if i > 5:

    break
ds_series = tf.data.Dataset.from_generator(

    gen_series, 

    output_types=(tf.int32, tf.float32), 

    output_shapes=((), (None,)))



ds_series
ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=([], [None]))



ids, sequence_batch = next(iter(ds_series_batch))

print(ids.numpy())

print()

print(sequence_batch.numpy())


flowers = tf.keras.utils.get_file(

    'flower_photos',

    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',

    untar=True)
#Create the image.ImageDataGenerator



img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)



images, labels = next(img_gen.flow_from_directory(flowers))



print(images.dtype, images.shape)

print(labels.dtype, labels.shape)
ds = tf.data.Dataset.from_generator(

    img_gen.flow_from_directory, args=[flowers], 

    output_types=(tf.float32, tf.float32), 

    output_shapes=([32,256,256,3], [32,5])

)



ds
# Creates a dataset that reads all of the examples from two files.

fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")
dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])

dataset
raw_example = next(iter(dataset))

parsed = tf.train.Example.FromString(raw_example.numpy())



parsed.features.feature['image/text']


inc_dataset = tf.data.Dataset.range(100)

dec_dataset = tf.data.Dataset.range(0, -100, -1)

dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))

batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):

  print([arr.numpy() for arr in batch])
#While tf.data tries to propagate shape information,

#the default settings of Dataset.batch result in an unknown batch size because the last batch may not be full. Note the Nones in the shape:

batched_dataset
batched_dataset = dataset.batch(7, drop_remainder=True)

batched_dataset
dataset = tf.data.Dataset.range(100)

dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))

dataset = dataset.padded_batch(4, padded_shapes=(None,))



for batch in dataset.take(2):

  print(batch.numpy())

  print()

 
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

titanic_lines = tf.data.TextLineDataset(titanic_file)



def plot_batch_sizes(ds):

  batch_sizes = [batch.shape[0] for batch in ds]

  plt.bar(range(len(batch_sizes)), batch_sizes)

  plt.xlabel('Batch number')

  plt.ylabel('Batch size')

titanic_batches = titanic_lines.repeat(3).batch(128)

plot_batch_sizes(titanic_batches)
#If you need clear epoch separation, put Dataset.batch before the repeat:



titanic_batches = titanic_lines.batch(128).repeat(3)



plot_batch_sizes(titanic_batches)
#If you would like to perform a custom computation (e.g. to collect statistics) at

#the end of each epoch then it's simplest to restart the dataset iteration on each epoch:



epochs = 3

dataset = titanic_lines.batch(128)



for epoch in range(epochs):

  for batch in dataset:

    print(batch.shape)

  print("End of epoch: ", epoch)
lines = tf.data.TextLineDataset(titanic_file)

counter = tf.data.experimental.Counter()



dataset = tf.data.Dataset.zip((counter, lines))

dataset = dataset.shuffle(buffer_size=100)

dataset = dataset.batch(20)

dataset
#Since the buffer_size is 100, and the batch size is 20, the first batch contains no elements with an index over 120.



n,line_batch = next(iter(dataset))

print(n.numpy())
dataset = tf.data.Dataset.zip((counter, lines))

shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)



print("Here are the item ID's near the epoch boundary:\n")

for n, line_batch in shuffled.skip(60).take(5):

  print(n.numpy())
shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]

plt.plot(shuffle_repeat, label="shuffle().repeat()")

plt.ylabel("Mean item ID")

plt.legend()
#But a repeat before a shuffle mixes the epoch boundaries together:



dataset = tf.data.Dataset.zip((counter, lines))

shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)



print("Here are the item ID's near the epoch boundary:\n")

for n, line_batch in shuffled.skip(55).take(15):

  print(n.numpy())
repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]



plt.plot(shuffle_repeat, label="shuffle().repeat()")

plt.plot(repeat_shuffle, label="repeat().shuffle()")

plt.ylabel("Mean item ID")

plt.legend()
range_ds = tf.data.Dataset.range(20)



iterator = iter(range_ds)

ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)

manager = tf.train.CheckpointManager(ckpt, '/tmp/my_ckpt', max_to_keep=3)



print([next(iterator).numpy() for _ in range(5)])



save_path = manager.save()



print([next(iterator).numpy() for _ in range(5)])



ckpt.restore(manager.latest_checkpoint)



print([next(iterator).numpy() for _ in range(5)])