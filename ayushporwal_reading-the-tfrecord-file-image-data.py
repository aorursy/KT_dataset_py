# Importing tensorflow and its helper to display images



import tensorflow as tf

import IPython.display as display
# Let's select one TFRecord and read it using tf.data.TFRecordDataset class in tensorflow



file_path = '../input/flower-classification-with-tpus/tfrecords-jpeg-192x192/train/00-192x192-798.tfrec'

raw_data = tf.data.TFRecordDataset(file_path)



# Now we have TFRecord data in the raw_data variable but to read and display images,

# we need to know with what features the image was encoded into this TFRecord file.



# So in the raw_data we will iterate over one example/image

for data in raw_data.take(1):

    example = tf.train.Example()

    example.ParseFromString(data.numpy())

    print(example)
# We see that it has three features namely id, class, and image of type int64, string, and string respectively.

# byte_list genric type can be coerced into string type

# Let's quickly make a feature dictionary and parse the raw_data using .map



feature_descp = {

    'class': tf.io.FixedLenFeature([], tf.int64),

    'image': tf.io.FixedLenFeature([], tf.string),

    'id': tf.io.FixedLenFeature([], tf.string),

}



def parse_example(example):

    return tf.io.parse_single_example(example, feature_descp)



parsed_data = raw_data.map(parse_example)



parsed_data
# In this step we can send the image data to TPU's or as I do here simply display images.

for example_image in parsed_data:

    images = example_image['image'].numpy()

    display.display(display.Image(data = images))