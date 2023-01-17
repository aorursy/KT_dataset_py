!git clone https://github.com/tensorflow/models.git

import sys
sys.path.append('/kaggle/working/models/research/object_detection/utils')
sys.path.append('/kaggle/working/models/research/object_detection/dataset_tools')
import tensorflow as tf

import dataset_util

import pandas as pd
import pydicom

from io import BytesIO

import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

debug = False

def create_tf_example(patientId, boxes):
    height = 1024 # Image height
    width = 1024 # Image width

    path = "../input/stage_1_train_images/" + patientId + ".dcm"

    ds = pydicom.dcmread(path)

    filename = bytes(patientId + '.jpg', 'utf-8') # Filename of the image. Empty if image is not from file
    image_format = b'jpeg' # b'jpeg' or b'png'

    encoded_image_data = ds.PixelData[16:]
    if (debug):
        print(encoded_image_data[:3])

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)

    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in boxes:
        if not np.isnan(box[0]):
            if (debug):
                print(box)
            classes_text.append(b'pneumonia')
            classes.append(1)
            
            # x-min y-min width height
            xmins.append(box[0] / width)   # store normalized values for bbox
            xmaxs.append((box[0] + box[2]) / width)
            ymins.append(box[1] / height)
            ymaxs.append((box[1] + box[3]) / height)

    if (debug):
        print(xmins)
        print(xmaxs)
        print(ymins)
        print(ymaxs)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

import contextlib2
import tf_record_creation_util

num_shards=10
output_filebase='train'

with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_filebase, num_shards)


    train = pd.read_csv("../input/stage_1_train_labels.csv")
    groups = train.groupby('patientId')

    count = 0

    for patientId in train.drop_duplicates('patientId')['patientId']:
        print('[{c}]processing patientId={p}'.format(c=count,p=patientId))

        boxes = groups.get_group(patientId).drop(columns=['patientId']).as_matrix()
        tf_example = create_tf_example(patientId, boxes)

        output_shard_index = count % num_shards
        output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

        count += 1

# clean up names a bit and take the first as validation
!mkdir train
!mkdir val

# TODO: use train_test_split to do validation rows
!mv train-00000-of-00010 val/val-00001-of-00001

!mv train-00001-of-00010 train/train-00001-of-00009
!mv train-00002-of-00010 train/train-00002-of-00009
!mv train-00003-of-00010 train/train-00003-of-00009
!mv train-00004-of-00010 train/train-00004-of-00009
!mv train-00005-of-00010 train/train-00005-of-00009
!mv train-00006-of-00010 train/train-00006-of-00009
!mv train-00007-of-00010 train/train-00007-of-00009
!mv train-00008-of-00010 train/train-00008-of-00009
!mv train-00009-of-00010 train/train-00009-of-00009
# remove the models git repo
!rm -rf models