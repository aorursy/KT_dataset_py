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
import tensorflow as tf
config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Get file path
video_record = "../input/video-sample/video/train00.tfrecord"
frame_record = "../input/frame-sample/frame/train00.tfrecord"
# read file and get record_iterator
record_iterator = tf.python_io.tf_record_iterator(video_record)
# Maybe many records in oen TFRecord file, we just need one record to analyze structur
record_0 = [record for record in record_iterator][0]
# parse record string as tf.train.Example
example = tf.train.Example.FromString(record_0)
# analyze structur
# pleas pay atention to 'key' and 'value' of feature ,especialy the data_type of the value'
print(example)
# Ther are two ways to get the value of record.
# First
example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
example.features.feature['labels'].int64_list.value
example.features.feature['mean_rgb'].float_list.value
example.features.feature['mean_audio'].float_list.value
example.features.feature['id'].bytes_list.value
# Second 
# encourge to use this way
feature_keys  = {'id': tf.FixedLenFeature([],tf.string),
                 'labels': tf.VarLenFeature(tf.int64),
                 'mean_rgb': tf.FixedLenFeature([1024],tf.float32),
                 'mean_audio': tf.FixedLenFeature([128],tf.float32)}

parsed = tf.parse_single_example(record_0,feature_keys)
# NOTE:: tf.VarLenFeature(tf.int64) will parse and return a sparse tensor, should cover it to dense tensor
# 3862:: according to YouTube-8M Tensorflow Starter Code, dataset have 3862 labels
parsed["labels"] = tf.sparse_to_dense(parsed["labels"].values, [3862], 1) 
sess.run(parsed)
# The function to parse record
def parser(record):
    feature_keys  = {'id': tf.FixedLenFeature([],tf.string),
                     'labels': tf.VarLenFeature(tf.int64),
                     'mean_rgb': tf.FixedLenFeature([1024],tf.float32),
                     'mean_audio': tf.FixedLenFeature([128],tf.float32)}                                                    
    parsed= tf.parse_single_example(record,feature_keys)
    parsed["labels"] = tf.sparse_to_dense(parsed["labels"].values, [3862], 1) 
    return parsed
# The tool
def input_video_data(video_path,batch_size=1,num_epoch=1):
    # Get all TFRecord files in this path
    # The first item of os.listdir(video_path) is current path
    filenames = [os.path.join(video_path,file) for file in os.listdir(video_path)][1:]
    # creat dataset
    dataset  = tf.data.TFRecordDataset(filenames)
    # parse every record string 
    dataset = dataset.map(parser)
    # Random shuffle
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epoch)
    iterator = dataset.make_one_shot_iterator()
    try:
        next_element = iterator.get_next()
    except tf.errors.OutOfRangeError:
        print("Iterations exhausted")
    return next_element
# Use this tool to get data
video_path = "../input/video-sample/video"
batch_example = input_video_data(video_path)
# every training time, sess.run will get a batch of data.
sess.run(batch_example)
record_iterator = tf.python_io.tf_record_iterator(frame_record)
record_0 = [record for record in record_iterator][100]
# NOTE：frame-level TFRecord files contain SequenceExamples
example = tf.train.SequenceExample.FromString(record_0)

# print(example)

# keys to parse context_features
feature_keys  = {'id': tf.FixedLenFeature([],tf.string),
                 'labels':tf.VarLenFeature(tf.int64)}
# keys to parse features_lists
sequence_features_keys = {'audio': tf.FixedLenSequenceFeature([],tf.string,allow_missing=True),
                          'rgb': tf.FixedLenSequenceFeature([],tf.string,allow_missing=True)}
# Use tf.parse_single_sequence_example to parse sequenceExample 
parsed = tf.parse_single_sequence_example(record_0,feature_keys,sequence_features_keys)
parsed[0]["labels"] = tf.sparse_to_dense(parsed[0]["labels"].values, [3862], 1) 
# return tuple:（features_dict，sequence_features_dict）
result = sess.run(parsed)
num_classes = 3862
def parser(record):
    feature_keys  = {'id': tf.FixedLenFeature([],tf.string),
                     'labels': tf.VarLenFeature(tf.int64)}
    sequence_features_keys = {'audio': tf.FixedLenSequenceFeature([],tf.string),
                              'rgb': tf.FixedLenSequenceFeature([],tf.string)}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(record,feature_keys,sequence_features_keys)
    context_parsed["labels"] = tf.sparse_to_dense(context_parsed["labels"].values, [num_classes], 1,validate_indices=False)
    return context_parsed, sequence_parsed

def input_frame_data(frame_path,batch_size=1,num_epoch=1):
    filenames = [os.path.join(frame_path,file) for file in os.listdir(frame_path)][1:]
    dataset  = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epoch)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element
frame_path = "../input/frame-sample/frame"
batch_example = input_frame_data(frame_path)
result = sess.run(batch_example)
# 1024 bit features
len(result[1]['rgb'][0][0])
# 300 up frames
len(result[1]['rgb'][0])