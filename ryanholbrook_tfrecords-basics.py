import tensorflow as tf

import numpy as np



PATH = '/kaggle/working/data.tfrecord'



with tf.io.TFRecordWriter(path=PATH) as f:

    f.write(b'123') # write one record

    f.write(b'xyz314') # write another record



with open(PATH, 'rb') as f:

    print(f.read())
x = tf.constant([[1, 2], [3, 4]], dtype=tf.uint8)

print('x:', x, '\n')



x_bytes = tf.io.serialize_tensor(x)

print('x_bytes:', x_bytes, '\n')



print('x:', tf.io.parse_tensor(x_bytes, out_type=tf.uint8))
from tensorflow.data import Dataset, TFRecordDataset

from tensorflow.data.experimental import TFRecordWriter



# Construct a small dataset

ds = Dataset.from_tensor_slices([b'abc', b'123'])



# Write the dataset to a TFRecord

writer = TFRecordWriter(PATH)

writer.write(ds)

    

# Read the dataset from the TFRecord

ds_2 = TFRecordDataset(PATH)

for x in ds_2:

    print(x)
# Create a dataset

features = tf.constant([

    [1, 2],

    [3, 4],

    [5, 6],

], dtype=tf.uint8)

ds = Dataset.from_tensor_slices(features)



# Serialize the tensors

ds_bytes = ds.map(tf.io.serialize_tensor)



# Write a TFRecord

writer = TFRecordWriter(PATH)

writer.write(ds_bytes)



# Read it back

ds_bytes_2 = TFRecordDataset(PATH)

ds_2 = ds_2.map(lambda x: tf.io.parse_tensor(x, out_type=tf.uint8))



# They are the same!

for x in ds:

    print(x)

print()

for x in ds_2:

    print(x)
from sklearn.datasets import load_sample_image

import matplotlib.pyplot as plt



# Load numpy array

image_raw = load_sample_image('flower.jpg')

print("Type {} with dtype {}".format(type(image_raw), image_raw.dtype))

plt.imshow(image_raw)

plt.title("Numpy")

plt.show()
from IPython.display import Image



# jpeg encode / decode

image_jpeg = tf.io.encode_jpeg(image_raw)

print("Type {} with dtype {}".format(type(image_jpeg), image_jpeg.dtype))

print("Sample: {}".format(image_jpeg.numpy()[:25]))

Image(image_jpeg.numpy())
image_raw_2 = tf.io.decode_jpeg(image_jpeg)



print("Type {} with dtype {}".format(type(image_raw_2), image_raw_2.dtype))

plt.imshow(image_raw_2)

plt.title("Numpy")

plt.show()
from tensorflow.train import BytesList, FloatList, Int64List

from tensorflow.train import Example, Features, Feature



# The Data

image = tf.constant([ # this could also be a numpy array

    [0, 1, 2],

    [3, 4, 5],

    [6, 7, 8],

])

label = 0

class_name = "Class A"





# Wrap with Feature as a BytesList, FloatList, or Int64List

image_feature = Feature(

    bytes_list=BytesList(value=[

        tf.io.serialize_tensor(image).numpy(),

    ])

)

label_feature = Feature(

    int64_list=Int64List(value=[label]),

)

class_name_feature = Feature(

    bytes_list=BytesList(value=[

        class_name.encode()

    ])

)





# Create a Features dictionary

features = Features(feature={

    'image': image_feature,

    'label': label_feature,

    'class_name': class_name_feature,

})



# Wrap with Example

example = Example(features=features)



print(example)
print(example.features.feature['label'])
example_bytes = example.SerializeToString()

print(example_bytes)
def make_example(image, label, class_name):

    image_feature = Feature(

        bytes_list=BytesList(value=[

            tf.io.serialize_tensor(image).numpy(),

        ])

    )

    label_feature = Feature(

        int64_list=Int64List(value=[

            label,

        ])

    )

    class_name_feature = Feature(

        bytes_list=BytesList(value=[

            class_name.encode(),

        ])

    )



    features = Features(feature={

        'image': image_feature,

        'label': label_feature,

        'class_name': class_name_feature,

    })

    

    example = Example(features=features)

    

    return example.SerializeToString()
example = make_example(

    image=np.array([[1, 2], [3, 4]]),

    label=1,

    class_name="Class B",

)



print(example)
from tensorflow.io import FixedLenFeature, VarLenFeature



feature_description = {

    'image': FixedLenFeature([], tf.string),

    'label': FixedLenFeature([], tf.int64),

    'class_name': FixedLenFeature([], tf.string),

}



example_2 = tf.io.parse_single_example(example, feature_description)

print("Parsed:   ", example_2)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset