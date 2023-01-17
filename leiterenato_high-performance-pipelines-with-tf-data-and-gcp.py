# First let's import Tensorflow
import tensorflow as tf
# Now import some additional libraries
from numpy import zeros
import numpy as np
from datetime import datetime
# Benchmark function for dataset
import time
default_timeit_steps = 1000
BATCH_SIZE = 1

# Iterate through each element of a dataset. An element is a pair 
# of image and label.
def timeit(ds: tf.data.TFRecordDataset, steps: int = default_timeit_steps, 
           batch_size: int = BATCH_SIZE) -> None:
    
    start = time.time()
    it = iter(ds)
    
    for i in range(steps):
        batch = next(it)
        
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()
    
    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size*steps/duration))
# Global variables

# Paths where images are located
FILENAMES = 'gs://tf-data-pipeline/*/*.jpg'

# Paths where labels can be parsed
FOLDERS = 'gs://tf-data-pipeline/*'

# Image resolution and shape
RESOLUTION = (224,224)
IMG_SHAPE=(224,224,3)

# tf.data AUTOTUNE
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Get labels from folder's name and create a map to an ID
def get_label_map(path: str) -> (dict, dict):
    #list folders in this path
    folders_name = tf.io.gfile.glob(path)

    labels = []
    for folder in folders_name:
        labels.append(folder.split(sep='/')[-1])

    # Generate a Label Map and Interted Label Map
    label_map = {labels[i]:i for i in range(len(labels))}
    inv_label_map = {i:labels[i] for i in range(len(labels))}
    
    return label_map, inv_label_map
# One hot encode the image's labels
def one_hot_encode(label_map: dict, filepath: list) -> dict:
    labels = dict()
    
    for i in range(len(filepath)):
        encoding = zeros(len(label_map), dtype='uint8')
        encoding[label_map[filepath[i].split(sep='/')[-2]]] = 1
        
        labels.update({filepath[i]:list(encoding)})
    
    return labels
label_map, inv_label_map = get_label_map(FOLDERS)
list(label_map.items())[:5]
# List all files in bucket
filepath = tf.io.gfile.glob(FILENAMES)
NUM_TOTAL_IMAGES = len(filepath)
# Split the features (image path) from labels
dataset = one_hot_encode(label_map, filepath)
dataset = [[k,v] for k,v in dataset.items()]

features = [i[0] for i in dataset]
labels = [i[1] for i in dataset]
# Create Dataset from Features and Labels
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# Example of one element of the dataset
# At this point we have a dataset containing the path and labels of an image
print(next(iter(dataset)))
# Download image bytes from Cloud Storage
def get_bytes_label(filepath, label):
    raw_bytes = tf.io.read_file(filepath)
    return raw_bytes, label
# Preprocess Image
def process_image(raw_bytes, label):
    image = tf.io.decode_jpeg(raw_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (224,224))
    
    return image, label
# Map transformations for each element inside the dataset
# Maps are separated as IO Intensive and CPU Intensive
def build_dataset(dataset, batch_size=BATCH_SIZE, cache=False):
    
    dataset = dataset.shuffle(NUM_TOTAL_IMAGES)
    
    # Extraction: IO Intensive
    dataset = dataset.map(get_bytes_label, num_parallel_calls=AUTOTUNE)

    # Transformation: CPU Intensive
    dataset = dataset.map(process_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(filename=cache)
        else:
            dataset = dataset.cache()
    
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
# Apply transformations to the dataset with images paths and labels
train_ds = build_dataset(dataset)
local_ds = train_ds.take(1).cache().repeat()
timeit(local_ds, 20000, batch_size=1)
# Iterate through this dataset for 1000 steps.
timeit(train_ds, batch_size=1, steps=1000)
# Memory
train_cache_ds = build_dataset(dataset, cache=True)
timeit(train_cache_ds, batch_size=1, steps=50000)
# Local Cache File
train_local_cache_ds = build_dataset(dataset, cache='./dog.tfcache', batch_size=1)
timeit(train_local_cache_ds, batch_size=1, steps=50000)
tf.summary.trace_off()
tf.summary.trace_on(graph=False, profiler=True)

train_ds = build_dataset(dataset)
timeit(train_ds, steps=1000)

tf.summary.trace_export('Data Pipeline', profiler_outdir='/home/jupyter/tensorflow-data-pipeline/logs/')
# Load the TensorBoard notebook extension.
%load_ext tensorboard
# Start tensorboard inside one cell
%tensorboard --logdir=/home/jupyter/tensorflow-data-pipeline/logs
# Function to download bytes from Cloud Storage
def get_bytes_label_tfrecord(filepath, label):
    raw_bytes = tf.io.read_file(filepath)
    return raw_bytes, label
# Preprocess Image
def process_image_tfrecord(raw_bytes, label):
    image = tf.io.decode_jpeg(raw_bytes, channels=3)
    image = tf.image.resize(image, (224,224), method='nearest')
    image = tf.io.encode_jpeg(image, optimize_size=True)
    
    return image, label
# Read images, preprocess and return a dataset
def build_dataset_tfrecord(dataset):
    
    dataset = dataset.map(get_bytes_label_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)
    
    return dataset
def tf_serialize_example(image, label):
    
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
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))    
    
    def serialize_example(image, label):
        
        feature = {
            'image': _bytes_feature(image),
            'label': _int64_feature(label)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        
        return example_proto.SerializeToString()
    
    tf_string = serialize_example(image, label)

    return tf_string
# Create TFRecord with `n_shards` shards
def create_tfrecord(ds, n_shards):

    for i in range(n_shards):
        batch = map(lambda x: tf_serialize_example(x[0],x[1]), ds.shard(n_shards, i)
                    .apply(build_dataset_tfrecord)
                    .as_numpy_iterator())
        
        with tf.io.TFRecordWriter('output_file-part-{i}.tfrecord'.format(i=i), 'GZIP') as writer:
            print('Creating TFRecord ... output_file-part-{i}.tfrecord'.format(i=i))
            for a in batch:
                writer.write(a)
# We sharded into 4 files with 130MB each.
# If the dataset is bigger, you can create more shards
create_tfrecord(dataset, 4)
TFRECORDS = 'gs://renatoleite-nb/tfrecords/*'
# Create a description of the features.
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
}
@tf.function
def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)
# List all the TFRecords and create a dataset from it
filenames = tf.io.gfile.glob(TFRECORDS)
filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
# Preprocess Image
@tf.function
def process_image_tfrecord(record):  
    image = tf.io.decode_jpeg(record['image'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    label = record['label']
    
    return image, label
# Create a Dataset composed of TFRecords (paths to bucket)
@tf.function
def get_tfrecord(filename):
    return tf.data.TFRecordDataset(filename, compression_type='GZIP', num_parallel_reads=AUTOTUNE)
def build_dataset_test(dataset, batch_size=BATCH_SIZE):
    
    dataset = dataset.interleave(get_tfrecord, num_parallel_calls=AUTOTUNE)
    
    # Transformation: IO Intensive 
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)

    # Transformation: CPU Intensive
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
test_ds = build_dataset_test(filenames_dataset, batch_size=32)
timeit(test_ds, steps=20000, batch_size=32)
def build_dataset_test(dataset, batch_size=BATCH_SIZE):
    
    dataset = dataset.interleave(get_tfrecord, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(process_image_tfrecord, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()
    # Pipeline next iteration
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
@tf.function
def _parse_function(example_proto):
    
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_example(example_proto, feature_description)
# Preprocess Image
@tf.function
def process_image_tfrecord(record):
    
    image = tf.map_fn(tf.io.decode_jpeg, record['image'], dtype=tf.uint8)
    image = tf.map_fn(lambda image: 
                      tf.image.convert_image_dtype(image, dtype=tf.float32), image, dtype=tf.float32)
    
    label = record['label']
    
    return image, label
test_ds = build_dataset_test(filenames_dataset, batch_size=32)
timeit(test_ds, steps=20000, batch_size=32)