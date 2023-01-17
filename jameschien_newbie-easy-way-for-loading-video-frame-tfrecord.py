import pandas as pd
import numpy as np
import tensorflow as tf
import os
path_label_dict = os.path.join('../input', 'label_names_2018.csv')

path_video_train = os.path.join('../input', 'video-sample', 'video', 'train00.tfrecord')
path_video_test = os.path.join('../input', 'video-sample', 'video', 'train01.tfrecord')

path_frame_train = os.path.join('../input', 'frame-sample', 'frame', 'train00.tfrecord')
path_frame_test = os.path.join('../input', 'frame-sample', 'frame', 'train01.tfrecord')
dfLabel = pd.read_csv(path_label_dict)
num_labels = len(dfLabel.label_name.unique())
def tfRecord_parse(record, num_labels, train=False):
    features = {
        'mean_rgb': tf.FixedLenFeature([1024], tf.float32),
        'mean_audio': tf.FixedLenFeature([128], tf.float32)
    }
    if train:
        features['labels'] = tf.VarLenFeature(tf.int64)
    parsed = tf.parse_single_example(record, features)
    x = tf.concat([parsed['mean_rgb'], parsed['mean_audio']], axis=0)
    if train:
        y = tf.sparse_to_dense(parsed['labels'].values, [num_labels], 1)
        return x, y
    return x
def tfRecord_seq_parse(record, num_labels, train=False):
    sequence_features = {
        'rgb': tf.FixedLenSequenceFeature([], tf.string),
        'audio': tf.FixedLenSequenceFeature([], tf.string)
    }
    context_features = {}
    if train:
        context_features['labels'] = tf.VarLenFeature(tf.int64)
    ctx, parsed = tf.parse_single_sequence_example(record, context_features=context_features, sequence_features=sequence_features)
    
    decode_seq_rgb = tf.decode_raw(parsed['rgb'], tf.uint8)
    decode_seq_rgb = tf.reshape(decode_seq_rgb, [-1, 1024])
    decode_seq_rgb = tf.cast(decode_seq_rgb, dtype=tf.float32)
    
    decode_seq_audio = tf.decode_raw(parsed['audio'], tf.uint8)
    decode_seq_audio = tf.reshape(decode_seq_audio, [-1, 128])
    decode_seq_audio = tf.cast(decode_seq_audio, dtype=tf.float32)
    
    x = tf.concat([decode_seq_rgb, decode_seq_audio], axis=1)
    if train:
        y = tf.sparse_to_dense(ctx['labels'].values, [num_labels], 1)
        return x, y
    return x
def generate(path, num_labels, batch_size=1, train=False, isFrame=False, num_parallel_calls=12):
    dataset = tf.data.TFRecordDataset(path)
    if isFrame:
        dataset = dataset.map(map_func=lambda x: tfRecord_seq_parse(x, num_labels, train=train), num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.map(map_func=lambda x: tfRecord_parse(x, num_labels, train=train), num_parallel_calls=num_parallel_calls)
    dataset = dataset.repeat(1000)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    
    return dataset.make_one_shot_iterator()
video_train = generate(path_video_train, num_labels, train=True)
frame_train = generate(path_frame_train, num_labels, train=True, isFrame=True)
next_video_val = video_train.get_next()
next_frame_val = frame_train.get_next()
with tf.Session() as session:
    x, y = session.run(next_video_val)
print('x: {}, y: {}'.format(x.shape, y.shape))
with tf.Session() as session:
    x, y = session.run(next_frame_val)
print('x: {}, y: {}'.format(x.shape, y.shape))