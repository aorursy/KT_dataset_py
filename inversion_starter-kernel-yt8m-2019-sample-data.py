import numpy as np 

import pandas as pd



import os

print(os.listdir("../input"))
print(os.listdir("../input/frame-sample/frame"))
# Loading libraries & datasets

import tensorflow as tf

import numpy as np

from IPython.display import YouTubeVideo



frame_lvl_record = "../input/frame-sample/frame/train00.tfrecord"
vid_ids = []

labels = []



for example in tf.python_io.tf_record_iterator(frame_lvl_record):

    tf_example = tf.train.Example.FromString(example)

    vid_ids.append(tf_example.features.feature['id']

                   .bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)
print('Number of videos in this tfrecord: ',len(vid_ids))

print('Picking a youtube video id:',vid_ids[13])
# With that video id, we can play the video

YouTubeVideo('UzXQaOLQVCU')
# due to execution time, we're only going to read the first video



feat_rgb = []

feat_audio = []



for example in tf.python_io.tf_record_iterator(frame_lvl_record):  

    tf_seq_example = tf.train.SequenceExample.FromString(example)

    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

    sess = tf.InteractiveSession()

    rgb_frame = []

    audio_frame = []

    # iterate through frames

    for i in range(n_frames):

        rgb_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['rgb']

                  .feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        audio_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['audio']

                  .feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        

        

    sess.close()

    

    feat_audio.append(audio_frame)

    feat_rgb.append(rgb_frame)

    break
print('The first video has %d frames' %len(feat_rgb[0]))
from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.manifold import TSNE

import numpy as np
vocab = pd.read_csv('../input/vocabulary.csv')

print("we have {} unique labels in the dataset".format(len(vocab['Index'].unique())))
n = 30 # although, we'll only show those that appear in the 1,000 for this competition

from collections import Counter



label_mapping =  vocab[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']



top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)

top_n_labels = [int(i[0]) for i in top_n]

top_n_label_names = [label_mapping[x] for x in top_n_labels if x in label_mapping] # filter out the labels that aren't in the 1,000 used for this competition

print(top_n_label_names)
import networkx as nx

from itertools import combinations



G = nx.Graph()



G.clear()

for list_of_nodes in labels:

    filtered_nodes = set(list_of_nodes).intersection(set(top_n_labels) & 

                                                     set(vocab['Index'].unique()))  

    for node1,node2 in list(combinations(filtered_nodes,2)): 

        node1_name = label_mapping[node1]

        node2_name = label_mapping[node2]

        G.add_node(node1_name)

        G.add_node(node2_name)

        G.add_edge(node1_name, node2_name)



plt.figure(figsize=(9,9))

nx.draw_networkx(G, font_size="12")
validate_record = "../input/validate-sample/validate/validate00.tfrecord"
val_vid_ids = []

val_vid_labels = []

segment_start_times = []

segment_end_times = []

segment_labels = []

segment_scores = []



for example in tf.python_io.tf_record_iterator(validate_record):

    tf_example = tf.train.Example.FromString(example)

    val_vid_ids.append(tf_example.features.feature['id']

                   .bytes_list.value[0].decode(encoding='UTF-8'))

    val_vid_labels.append(tf_example.features.feature['labels'].int64_list.value)

    segment_start_times.append(tf_example.features.feature['segment_start_times'].int64_list.value)

    segment_end_times.append(tf_example.features.feature['segment_end_times'].int64_list.value)

    segment_labels.append(tf_example.features.feature['segment_labels'].int64_list.value)

    segment_scores.append(tf_example.features.feature['segment_scores'].float_list.value)

    
print(val_vid_ids[0])

print(val_vid_labels[0])

print(segment_start_times[0])

print(segment_end_times[0])

print(segment_labels[0])

print(segment_scores[0])