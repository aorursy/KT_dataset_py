import math, re, os

import cv2

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from google.cloud import storage

print(f"Tensor Flow version: {tf.__version__}")

AUTO = tf.data.experimental.AUTOTUNE
PATH = "/kaggle/input/applications-of-deep-learning-wustl-fall-2020/final-kaggle-data/"



PATH_TRAIN = os.path.join(PATH, "train.csv")

PATH_TEST = os.path.join(PATH, "test.csv")
# load the meta data

df_train = pd.read_csv(PATH_TRAIN)

df_test = pd.read_csv(PATH_TEST)



# remove image 1300

removal_mask = df_train["id"] != 1300 # corrupt image

df_train = df_train[removal_mask]



df_train["filename"] = df_train["id"].astype(str) + ".png"

df_train["stable"] = df_train["stable"].astype(int)



df_test["filename"] = df_test["id"].astype(str) + ".png"



df_train.info()
IMGS = df_train["filename"].to_list()

len(IMGS)
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
def serialize_train_example(feature0, feature1, feature2, feature3):

    feature = {

        'image': _bytes_feature(feature0),

        'id': _int64_feature(feature1),

        'filename': _bytes_feature(feature2),

        'stable': _int64_feature(feature3)

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
dims = [ # output image sizes

    (192, 192),

    (331, 331),

    ]
os.mkdir('./Train')



for dim in dims:

    SIZE = 2071

    OUT_PATH = './Train/%ix%i'%(dim[0],dim[1])

    os.mkdir(OUT_PATH)

    CT = len(IMGS) // SIZE + int(len(IMGS) % SIZE != 0)

    for j in range(CT):

        print()

        print('Writing TFRecord %i of %i...'%(j,CT))

        CT2 = min(SIZE, len(IMGS) - j * SIZE)

        with tf.io.TFRecordWriter(OUT_PATH + '/train%.2i-%i.tfrec'%(j,CT2)) as writer:

            for k in range(CT2):

                img = cv2.imread(PATH + IMGS[SIZE * j + k])

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # fix incorrect colors

                # reshape image to square

                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)           

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring() # maybe change to '.png'?

                filename = IMGS[SIZE * j + k] # .split('.')[0]

                row = df_train[df_train.filename == filename]

                example = serialize_train_example(

                    img,

                    row.id.values[0],

                    str.encode(filename),

                    row.stable.values[0]

                )

                writer.write(example)

                if k % 100 == 0:

                    print(k, ',  ',end='')
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)

CLASSES = [0,1]



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    #if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

    #    numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(

        CLASSES[label], 

        'OK' if correct else 'NO', 

        u"\u2192" if not correct else '',

        CLASSES[correct_label] if not correct else ''

    ), correct



def display_one_image(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(

            title, 

            fontsize=int(titlesize) if not red else int(titlesize/1.2), 

            color='red' if red else 'black', 

            fontdict={'verticalalignment':'center'}, 

            pad=int(titlesize/1.5)

        )

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

    # auto-squaring: this will drop data that doesnt fit into a square subplot

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot = (rows,cols,1)

    if rows < cols:

        plt.figure(figsize = (FIGSIZE, FIGSIZE / cols * rows))

    else:

        plt.figure(figsize = (FIGSIZE / cols * rows, FIGSIZE))

    #display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = label

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3

        subplot = display_one_image(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0 # converts image to float in [0,1]

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "filename": tf.io.FixedLenFeature([], tf.string), # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example["image"])

    label = example["filename"]

    return image, label



def load_dataset(filenames, labeled=True, ordered=False):

    """Read from TFRecords. For optimal performance, reading from multiple 

    files at once and disregarding data order. Order does not matter since 

    we will be shuffling the data anyway"""

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order

    

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_labeled_tfrecord)

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
# Initialise variables

HEIGHT, WIDTH = dims[0] # look at the first image size

IMAGE_SIZE = [HEIGHT, WIDTH]

BATCH_SIZE = 32

AUTO = tf.data.experimental.AUTOTUNE

OUT_PATH = './Train/%ix%i'%(HEIGHT, WIDTH)

TRAINING_FILENAMES = tf.io.gfile.glob(OUT_PATH + '/train*.tfrec')

print('There are %i train images'%count_data_items(TRAINING_FILENAMES))
#Display train images

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)

display_batch_of_images(next(train_batch))
TEST_IMGS = df_test["filename"].to_list()

len(TEST_IMGS)
def serialize_test_example(feature0, feature1, feature2):

    feature = {

        'image': _bytes_feature(feature0),

        'id': _int64_feature(feature1),

        'filename': _bytes_feature(feature2),

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
os.mkdir('./Test')



for dim in dims:

    SIZE = 2071

    OUT_PATH = './Test/%ix%i'%(dim[0],dim[1])

    os.mkdir(OUT_PATH)

    CT = len(TEST_IMGS) // SIZE + int(len(TEST_IMGS) % SIZE != 0)

    for j in range(CT):

        print()

        print('Writing TFRecord %i of %i...'%(j,CT))

        CT2 = min(SIZE, len(TEST_IMGS) - j * SIZE)

        with tf.io.TFRecordWriter(OUT_PATH + '/test%.2i-%i.tfrec'%(j,CT2)) as writer:

            for k in range(CT2):

                img = cv2.imread(PATH + TEST_IMGS[SIZE * j + k])

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # fix incorrect colors

                # reshape image to square

                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)           

                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring() # maybe change to '.png'?

                filename = TEST_IMGS[SIZE * j + k] # .split('.')[0]

                row = df_test[df_test.filename == filename]

                example = serialize_test_example(

                    img,

                    row.id.values[0],

                    str.encode(filename)

                )

                writer.write(example)

                if k % 100 == 0:

                    print(k, ',  ',end='')