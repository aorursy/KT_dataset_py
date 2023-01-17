import pathlib

import os

import io

from collections import namedtuple, OrderedDict

import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf

import keras

import IPython.display as display

from PIL import Image

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
path = '../input/playing-cards-dataset'

path_train = os.path.join(path,'train_zipped')

path_test = os.path.join(path,'test_zipped')

print(path_train)



train = pd.read_csv(os.path.join(path,'train_cards_label.csv'))

test = pd.read_csv(os.path.join(path,'test_cards_label.csv'))

print(train.shape)

print(test.shape)

print('_' * 49)

print(train.head())

print('_' * 100)

print(test.head())

print(train['class'].unique())
def display_image(i):

    row = train.filename.unique()

    try:

        img = cv2.imread(os.path.join(path_train, row[i]))[...,::-1]

        print(img.shape)

        plt.axis("off")

        plt.imshow(img)

        plt.show()

    except:

        print('out of bound')
display_image(10)
IMG_SIZE = 500

CHANNELS = 3

BATCH_SIZE = 256

SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

N_LABELS = 52
AUTOTUNE = tf.data.experimental.AUTOTUNE

list_ds = tf.data.Dataset.list_files(str(path_train+'/*.jpg'))

for f in list_ds.take(5):

    print(f.numpy())
def recursive_parse_xml_to_dict(xml):

    """Recursively parses XML contents to python dict."""

    if not xml:

        return {xml.tag: xml.text}

    result = {}

    for child in xml:

        child_result = recursive_parse_xml_to_dict(child)

        if child.tag != 'object':

            result[child.tag] = child_result[child.tag]

        else:

            if child.tag not in result:

                result[child.tag] = []

            result[child.tag].append(child_result[child.tag])

    return {xml.tag: result}



def read_examples_list(path):

    """Read list of training or validation examples."""

    with tf.gfile.GFile(path) as fid:

        lines = fid.readlines()

    return [line.strip().split(' ')[0] for line in lines]
def class_text_to_int(row_label):

    if row_label == '2s':

        return 33

    elif row_label == '2c':

        return 34

    elif row_label == '2d':

        return 35

    elif row_label == '2h':

        return 36

    elif row_label == '3s':

        return 37

    elif row_label == '3c':

        return 38

    elif row_label == '3d':

        return 39

    elif row_label == '3h':

        return 40

    elif row_label == '4s':

        return 41

    elif row_label == '4c':

        return 42

    elif row_label == '4d':

        return 43

    elif row_label == '4h':

        return 44

    elif row_label == '5s':

        return 45

    elif row_label == '5c':

        return 46

    elif row_label == '5d':

        return 47

    elif row_label == '5h':

        return 48

    elif row_label == '6s':

        return 49

    elif row_label == '6c':

        return 50

    elif row_label == '6d':

        return 51

    elif row_label == '6h':

        return 52

    elif row_label == '7s':

        return 1

    elif row_label == '8s':

        return 2

    elif row_label == '9s':

        return 3

    elif row_label == 'Qs':

        return 4

    elif row_label == 'Ks':

        return 5

    elif row_label == '10s':

        return 6

    elif row_label == 'As':

        return 7

    elif row_label == 'Js':

        return 8

    elif row_label == '7h':

        return 9

    elif row_label == '8h':

        return 10

    elif row_label == '9h':

        return 11

    elif row_label == 'Qh':

        return 12

    elif row_label == 'Kh':

        return 13

    elif row_label == '10h':

        return 14

    elif row_label == 'Ah':

        return 15

    elif row_label == 'Jh':

        return 16

    elif row_label == '7d':

        return 17

    elif row_label == '8d':

        return 18

    elif row_label == '9d':

        return 19

    elif row_label == 'Qd':

        return 20

    elif row_label == 'Kd':

        return 21

    elif row_label == '10d':

        return 22

    elif row_label == 'Ad':

        return 23

    elif row_label == 'Jd':

        return 24

    elif row_label == '7c':

        return 25

    elif row_label == '8c':

        return 26

    elif row_label == '9c':

        return 27

    elif row_label == 'Qc':

        return 28

    elif row_label == 'Kc':

        return 29

    elif row_label == '10c':

        return 30

    elif row_label == 'Ac':

        return 31

    elif row_label == 'Jc':

        return 32

    else:

        return 0

    

def split(df, group):

    data = namedtuple('data', ['filename', 'object'])

    gb = df.groupby(group)

    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]



def create_tf_example(group, path):

    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:

        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)

    width, height = image.size

    filename = group.filename.encode('utf8')

    image_format = b'jpg'

    xmins = []

    xmaxs = []

    ymins = []

    ymaxs = []

    classes_text = []

    classes = []

    for index, row in group.object.iterrows():

        xmins.append(row['xmin'] / width)

        xmaxs.append(row['xmax'] / width)

        ymins.append(row['ymin'] / height)

        ymaxs.append(row['ymax'] / height)

        classes_text.append(row['class'].encode('utf8'))

        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={

        'image/height': int64_feature(height),

        'image/width': int64_feature(width),

        'image/filename': bytes_feature(filename),

        'image/source_id': bytes_feature(filename),

        'image/encoded': bytes_feature(encoded_jpg),

        'image/format': bytes_feature(image_format),

        'image/object/bbox/xmin': float_list_feature(xmins),

        'image/object/bbox/xmax': float_list_feature(xmaxs),

        'image/object/bbox/ymin': float_list_feature(ymins),

        'image/object/bbox/ymax': float_list_feature(ymaxs),

        'image/object/class/text': bytes_list_feature(classes_text),

        'image/object/class/label': int64_list_feature(classes),

    }))

    return tf_example
filename_train = [os.path.join(path,'train.record')]

filename_test = [os.path.join(path,'test.record')]



train_data = tf.data.TFRecordDataset(filename_train)

test_data = tf.data.TFRecordDataset(filename_test)
for data in train_data.take(1):

    example = tf.train.Example()

    example.ParseFromString(data.numpy())

    print(example)