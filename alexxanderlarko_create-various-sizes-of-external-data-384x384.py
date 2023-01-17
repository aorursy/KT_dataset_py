# RESIZE_TO = (768, 768)

RESIZE_TO = (384, 384)

# RESIZE_TO = (256, 256)



if RESIZE_TO == (768, 768):

    JPEG_QUALITY = 85

elif  RESIZE_TO == (384, 384):

    JPEG_QUALITY = 98

else:

    JPEG_QUALITY = 100

    

print("RESIZE_TO: {0}".format(RESIZE_TO))

print("JPEG_QUALITY: {0}%".format(JPEG_QUALITY))
import numpy as np

import cv2

import os

import tensorflow as tf

import re

import math

import matplotlib.pyplot as plt
print(os.listdir('../input'))
print(tf.__version__)
from kaggle_datasets import KaggleDatasets



# you can list the bucket with "!gsutil ls $GCS_DS_PATH"

GCS_DS_PATH = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/test*.tfrec')
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        "patient_id": tf.io.FixedLenFeature([], tf.int64),

        "sex": tf.io.FixedLenFeature([], tf.int64),

        "age_approx": tf.io.FixedLenFeature([], tf.int64),

        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),

        "source": tf.io.FixedLenFeature([], tf.int64),

        "target": tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    image_name = example['image_name']

    patient_id = example['patient_id']

    sex = example['sex']

    age_approx = example['age_approx']

    anatom_site_general_challenge = example['anatom_site_general_challenge']

    source = example['source']

    target = example['target']

    # returns an image and features

    return image, (image_name, patient_id, sex, age_approx, \

            anatom_site_general_challenge, source, target)



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        "patient_id": tf.io.FixedLenFeature([], tf.int64),

        "sex": tf.io.FixedLenFeature([], tf.int64),

        "age_approx": tf.io.FixedLenFeature([], tf.int64),

        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),

        # no 'source' and 'target' for the test data.

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    image_name = example['image_name']

    patient_id = example['patient_id']

    sex = example['sex']

    age_approx = example['age_approx']

    anatom_site_general_challenge = example['anatom_site_general_challenge']

    # returns an image and features

    return image, (image_name, patient_id, sex, age_approx, \

            anatom_site_general_challenge)



def load_dataset(filenames, labeled=True):

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=None)

    read_tfrecord = read_labeled_tfrecord if labeled else read_unlabeled_tfrecord

    dataset = dataset.map(read_tfrecord, num_parallel_calls=None)

    # returns a dataset of (image, features) pairs

    return dataset
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0.0))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_train_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7):

    feature = {

        'image': _bytes_feature(feature0),

        'image_name': _bytes_feature(feature1),

        'patient_id': _int64_feature(feature2),

        'sex': _int64_feature(feature3),

        'age_approx': _int64_feature(feature4),

        'anatom_site_general_challenge': _int64_feature(feature5),

        'source': _int64_feature(feature6),

        'target': _int64_feature(feature7)

    }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
def serialize_test_example(feature0, feature1, feature2, feature3, feature4, feature5): 

    feature = {

        'image': _bytes_feature(feature0),

        'image_name': _bytes_feature(feature1),

        'patient_id': _int64_feature(feature2),

        'sex': _int64_feature(feature3),

        'age_approx': _int64_feature(feature4),

        'anatom_site_general_challenge': _int64_feature(feature5),

    }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
def write_tf_records(file_paths, labeled, serialize_example):

    for file_index, file_path in enumerate(file_paths):

        print()

        print('Writing TFRecord %i of %i...' % (file_index + 1, len(file_paths)))

        file_name = os.path.basename(file_path)

        with tf.io.TFRecordWriter(file_name) as writer:

            dataset = load_dataset(file_path, labeled)

            for data_index, (image_tensor, features) in enumerate(iter(dataset)):

                image_np = image_tensor.numpy()

                resized_image = cv2.resize(image_np, RESIZE_TO, interpolation=cv2.INTER_AREA)

                fixed_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR) # Fix incorrect colors

                jpg_image = cv2.imencode('.jpg', fixed_image, (cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY))[1].tostring()

                example = serialize_example(jpg_image, *features)

                writer.write(example)

                if data_index % 100 == 0:

                    print(data_index, ', ', end='')
write_tf_records(TRAINING_FILENAMES, True, serialize_train_example)
write_tf_records(TEST_FILENAMES, False, serialize_test_example)
! ls -l
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)

CLASSES = [0,1]



def batch_to_numpy_images_and_labels(data):

    images, features = data

    numpy_images = images.numpy()

    numpy_labels = features[0].numpy()

    if 7 <= len(features):

        numpy_targets = features[6].numpy()

        numpy_labels = [ "{0}: {1}".format(lbl, tgt) for lbl, tgt in zip(numpy_labels, numpy_targets) ]

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = label

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
RESIZED_TRAINING_FILENAMES = tf.io.gfile.glob('train*.tfrec')

print('There are %i train images' % count_data_items(RESIZED_TRAINING_FILENAMES))
# DISPLAY TRAIN IMAGES

training_dataset = load_dataset(RESIZED_TRAINING_FILENAMES, labeled=True)

training_dataset = training_dataset.batch(20)

train_batch = iter(training_dataset)



display_batch_of_images(next(train_batch))
RESIZED_TEST_FILENAMES = tf.io.gfile.glob('test*.tfrec')

print('There are %i test images' % count_data_items(RESIZED_TEST_FILENAMES))
# DISPLAY TEST IMAGES

test_dataset = load_dataset(RESIZED_TEST_FILENAMES, labeled=False)

test_dataset = test_dataset.batch(20)

test_batch = iter(test_dataset)



display_batch_of_images(next(test_batch))