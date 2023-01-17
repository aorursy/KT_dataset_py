import tensorflow as tf



tf.executing_eagerly()


import pathlib

import os

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import time as time



class ArtificialDataset(tf.data.Dataset):

    def _generator(num_samples):

        # Opening the file

        time.sleep(0.03)

        

        for sample_idx in range(num_samples):

            # Reading data (line, record) from the file

            time.sleep(0.015)

            

            yield (sample_idx,)

    

    def __new__(cls, num_samples=3):

        return tf.data.Dataset.from_generator(

            cls._generator,

            output_types=tf.dtypes.int64,

            output_shapes=(1,),

            args=(num_samples,)

        )
def benchmark(dataset, num_epochs=2):

    start_time = time.perf_counter()

    for epoch_num in range(num_epochs):

        for sample in dataset:

            # Performing a training step

            time.sleep(0.01)

    tf.print("Execution time:", time.perf_counter() - start_time)
benchmark(ArtificialDataset())
benchmark(

    ArtificialDataset()

    .prefetch(tf.data.experimental.AUTOTUNE)

)


benchmark(

    tf.data.Dataset.range(2)

    .interleave(ArtificialDataset)

)
benchmark(

    tf.data.Dataset.range(2)

    .interleave(

        ArtificialDataset,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

)
def mapped_function(s):

    # Do some hard pre-processing

    tf.py_function(lambda: time.sleep(0.03), [], ())

    return s
benchmark(

    ArtificialDataset()

    .map(mapped_function)

)
benchmark(

    ArtificialDataset()

    .map(

        mapped_function,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

)
benchmark(

    ArtificialDataset()

    .map(  # Apply time consuming operations before cache

        mapped_function

    ).cache(

    ),

    5

)
fast_dataset = tf.data.Dataset.range(10000)



def fast_benchmark(dataset, num_epochs=2):

    start_time = time.perf_counter()

    for _ in tf.data.Dataset.range(num_epochs):

        for _ in dataset:

            pass

    tf.print("Execution time:", time.perf_counter() - start_time)

    

def increment(x):

    return x+1
fast_benchmark(

    fast_dataset

    # Apply function one item at a time

    .map(increment)

    # Batch

    .batch(256)

)
fast_benchmark(

    fast_dataset

    .batch(256)

    # Apply function on a batch of items

    # The tf.Tensor.__add__ method already handle batches

    .map(increment)

)

#dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)
import itertools

from collections import defaultdict



import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt
class TimeMeasuredDataset(tf.data.Dataset):

    # OUTPUT: (steps, timings, counters)

    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)

    OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))

    

    _INSTANCES_COUNTER = itertools.count()  # Number of datasets generated

    _EPOCHS_COUNTER = defaultdict(itertools.count)  # Number of epochs done for each dataset

    

    def _generator(instance_idx, num_samples):

        epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])

        

        # Opening the file

        open_enter = time.perf_counter()

        time.sleep(0.03)

        open_elapsed = time.perf_counter() - open_enter

        

        for sample_idx in range(num_samples):

            # Reading data (line, record) from the file

            read_enter = time.perf_counter()

            time.sleep(0.015)

            read_elapsed = time.perf_counter() - read_enter

            

            yield (

                [("Open",), ("Read",)],

                [(open_enter, open_elapsed), (read_enter, read_elapsed)],

                [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]

            )

            open_enter, open_elapsed = -1., -1.  # Negative values will be filtered

            

    

    def __new__(cls, num_samples=3):

        return tf.data.Dataset.from_generator(

            cls._generator,

            output_types=cls.OUTPUT_TYPES,

            output_shapes=cls.OUTPUT_SHAPES,

            args=(next(cls._INSTANCES_COUNTER), num_samples)

        )
def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):

    # Remove invalid entries (negative times, or empty steps) from the timelines

    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]

    steps = timeline['steps'][invalid_mask].numpy()

    times = timeline['times'][invalid_mask].numpy()

    values = timeline['values'][invalid_mask].numpy()

    

    # Get a set of different steps, ordered by the first time they are encountered

    step_ids, indices = np.stack(np.unique(steps, return_index=True))

    step_ids = step_ids[np.argsort(indices)]



    # Shift the starting time to 0 and compute the maximal time value

    min_time = times[:,0].min()

    times[:,0] = (times[:,0] - min_time)

    end = max(width, (times[:,0]+times[:,1]).max() + 0.01)

    

    cmap = mpl.cm.get_cmap("plasma")

    plt.close()

    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})

    fig.suptitle(title)

    fig.set_size_inches(17.0, len(step_ids))

    plt.xlim(-0.01, end)

    

    for i, step in enumerate(step_ids):

        step_name = step.decode()

        ax = axs[i]

        ax.set_ylabel(step_name)

        ax.set_ylim(0, 1)

        ax.set_yticks([])

        ax.set_xlabel("time (s)")

        ax.set_xticklabels([])

        ax.grid(which="both", axis="x", color="k", linestyle=":")

        

        # Get timings and annotation for the given step

        entries_mask = np.squeeze(steps==step)

        serie = np.unique(times[entries_mask], axis=0)

        annotations = values[entries_mask]

        

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)

        if annotate:

            for j, (start, width) in enumerate(serie):

                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])

                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,

                        horizontalalignment='left', verticalalignment='center')

    if save:

        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")


def timelined_benchmark(dataset, num_epochs=2):

    # Initialize accumulators

    steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)

    times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)

    values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)

    

    start_time = time.perf_counter()

    for epoch_num in range(num_epochs):

        epoch_enter = time.perf_counter()

        for (steps, times, values) in dataset:

            # Record dataset preparation informations

            steps_acc = tf.concat((steps_acc, steps), axis=0)

            times_acc = tf.concat((times_acc, times), axis=0)

            values_acc = tf.concat((values_acc, values), axis=0)

            

            # Simulate training time

            train_enter = time.perf_counter()

            time.sleep(0.01)

            train_elapsed = time.perf_counter() - train_enter

            

            # Record training informations

            steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)

            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)

            values_acc = tf.concat((values_acc, [values[-1]]), axis=0)

        

        epoch_elapsed = time.perf_counter() - epoch_enter

        # Record epoch informations

        steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)

        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)

        values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)

        time.sleep(0.001)

    

    tf.print("Execution time:", time.perf_counter() - start_time)

    return {"steps": steps_acc, "times": times_acc, "values": values_acc}
def map_decorator(func):

    def wrapper(steps, times, values):

        # Use a tf.py_function to prevent auto-graph from compiling the method

        return tf.py_function(

            func,

            inp=(steps, times, values),

            Tout=(steps.dtype, times.dtype, values.dtype)

        )

    return wrapper


_batch_map_num_items = 50



def dataset_generator_fun(*args):

    return TimeMeasuredDataset(num_samples=_batch_map_num_items)


@map_decorator

def naive_map(steps, times, values):

    map_enter = time.perf_counter()

    time.sleep(0.001)  # Time contumming step

    time.sleep(0.0001)  # Memory consumming step

    map_elapsed = time.perf_counter() - map_enter



    return (

        tf.concat((steps, [["Map"]]), axis=0),

        tf.concat((times, [[map_enter, map_elapsed]]), axis=0),

        tf.concat((values, [values[-1]]), axis=0)

    )



naive_timeline = timelined_benchmark(

    tf.data.Dataset.range(2)

    .flat_map(dataset_generator_fun)

    .map(naive_map)

    .batch(_batch_map_num_items, drop_remainder=True)

    .unbatch(),

    5

)
draw_timeline(naive_timeline, "Naive", 15)


@map_decorator

def time_consumming_map(steps, times, values):

    map_enter = time.perf_counter()

    time.sleep(0.001 * values.shape[0])  # Time contumming step

    map_elapsed = time.perf_counter() - map_enter



    return (

        tf.concat((steps, tf.tile([[["1st map"]]], [steps.shape[0], 1, 1])), axis=1),

        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),

        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)

    )





@map_decorator

def memory_consumming_map(steps, times, values):

    map_enter = time.perf_counter()

    time.sleep(0.0001 * values.shape[0])  # Memory consumming step

    map_elapsed = time.perf_counter() - map_enter



    # Use tf.tile to handle batch dimension

    return (

        tf.concat((steps, tf.tile([[["2nd map"]]], [steps.shape[0], 1, 1])), axis=1),

        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),

        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)

    )





optimized_timeline = timelined_benchmark(

    tf.data.Dataset.range(2)

    .interleave(  # Parallelize data reading

        dataset_generator_fun,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

    .batch(  # Vectorize your mapped function

        _batch_map_num_items,

        drop_remainder=True)

    .map(  # Parallelize map transformation

        time_consumming_map,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

    .cache()  # Cache data

    .map(  # Reduce memory usage

        memory_consumming_map,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    )

    .prefetch(  # Overlap producer and consumer works

        tf.data.experimental.AUTOTUNE

    )

    .unbatch(),

    5

)
draw_timeline(naive_timeline, "Naive", 15)
draw_timeline(optimized_timeline, "Optimized", 15)
# source data - numpy array

data = np.arange(10)

# create a dataset from numpy array

dataset = tf.data.Dataset.from_tensor_slices(data)
data = tf.arange(10)

dataset = tf.data.Dataset.from_tensors(data)
def generator():

  for i in range(10):

    yield 2*i

    

dataset = tf.data.Dataset.from_generator(generator, (tf.int32))
data = np.arange(10,40)

# create batches of 10

dataset = tf.data.Dataset.from_tensor_slices(data).batch(10)

# creates the iterator to consume the data 

iterator = dataset.make_one_shot_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
datax = np.arange(10,20)

datay = np.arange(11,21)

datasetx = tf.data.Dataset.from_tensor_slices(datax)

datasety = tf.data.Dataset.from_tensor_slices(datay)

dcombined = tf.data.Dataset.zip((datasetx, datasety)).batch(2)

iterator = dcombined.make_one_shot_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))

dataset = dataset.repeat(count=2)

iterator = dataset.make_one_shot_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
def map_fnc(x):

  return x*2;

data = np.arange(10)

dataset = tf.data.Dataset.from_tensor_slices(data)

dataset = dataset.map(map_fnc)

iterator = dataset.make_one_shot_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
data = np.arange(10,15)

#create the dataset

dataset = tf.data.Dataset.from_tensor_slices(data)

#create the iterator

iterator = dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:

  val = sess.run(next_element)

  print(val)
# define two placeholders to accept min and max value

min_val = tf.placeholder(tf.int32, shape=[])

max_val = tf.placeholder(tf.int32, shape=[])

data = tf.range(min_val, max_val)

dataset = tf.data.Dataset.from_tensor_slices(data)

iterator = dataset.make_initializable_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

  

  # initialize an iterator with range of values from 10 to 15

  sess.run(iterator.initializer, feed_dict={min_val:10, max_val:15})

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass

      

  # initialize an iterator with range of values from 1 to 10

  sess.run(iterator.initializer, feed_dict={min_val:1, max_val:10})

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
def map_fnc(ele):

  return ele*2

min_val = tf.placeholder(tf.int32, shape=[])

max_val = tf.placeholder(tf.int32, shape=[])

data = tf.range(min_val, max_val)

#Define separate datasets for training and validation

train_dataset =  tf.data.Dataset.from_tensor_slices(data)

val_dataset = tf.data.Dataset.from_tensor_slices(data).map(map_fnc)

#create an iterator 

iterator=tf.data.Iterator.from_structure(train_dataset.output_types    ,train_dataset.output_shapes)

train_initializer = iterator.make_initializer(train_dataset)

val_initializer = iterator.make_initializer(val_dataset)

next_ele = iterator.get_next()

with tf.Session() as sess:

  

  # initialize an iterator with range of values from 10 to 15

  sess.run(train_initializer, feed_dict={min_val:10, max_val:15})

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass

      

  # initialize an iterator with range of values from 1 to 10

  sess.run(val_initializer, feed_dict={min_val:1, max_val:10})

  try:

    while True:

      val = sess.run(next_ele)

      print(val)

  except tf.errors.OutOfRangeError:

    pass
def map_fnc(ele):

  return ele*2

min_val = tf.placeholder(tf.int32, shape=[])

max_val = tf.placeholder(tf.int32, shape=[])

data = tf.range(min_val, max_val)

train_dataset = tf.data.Dataset.from_tensor_slices(data)

val_dataset = tf.data.Dataset.from_tensor_slices(data).map(map_fnc)

train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types , train_dataset.output_shapes)

train_initializer = train_val_iterator.make_initializer(train_dataset)

val_initializer = train_val_iterator.make_initializer(val_dataset)

test_dataset = tf.data.Dataset.from_tensor_slices(tf.range(10,15))

test_iterator = test_dataset.make_one_shot_iterator()

handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

next_ele = iterator.get_next()

with tf.Session() as sess:

  

  train_val_handle = sess.run(train_val_iterator.string_handle())

  test_handle = sess.run(test_iterator.string_handle())

  

  # training

  sess.run(train_initializer, feed_dict={min_val:10, max_val:15})

  try:

    while True:

      val = sess.run(next_ele, feed_dict={handle:train_val_handle})

      print(val)

  except tf.errors.OutOfRangeError:

    pass

      

  #validation

  sess.run(val_initializer, feed_dict={min_val:1, max_val:10})

  try:

    while True:

      val = sess.run(next_ele, feed_dict={handle:train_val_handle})

      print(val)

  except tf.errors.OutOfRangeError:

    pass

  

  #testing

  try:

    while True:

      val = sess.run(next_ele, feed_dict={handle:test_handle})

      print(val)

  except tf.errors.OutOfRangeError:

    pass