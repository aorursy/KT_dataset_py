# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

from IPython.display import YouTubeVideo





video_lvl_record = "../input/video_level/train-1.tfrecord"

frame_lvl_record = "../input/frame_level/train-1.tfrecord"
import operator

video_count_dict = {}

audio_sum_dict = {}

rgb_sum_dict = {}



filenames = ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]

for filename in filenames:

    for example in tf.python_io.tf_record_iterator(filename):

        tf_example = tf.train.Example.FromString(example)



        mean_rgb = tf_example.features.feature['mean_rgb'].float_list.value

        average_rgb = sum(mean_rgb) / float(len(mean_rgb))



        mean_audio = tf_example.features.feature['mean_audio'].float_list.value

        average_audio = sum(mean_audio) / float(len(mean_audio))

        # vid_ids = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')

        labels = tf_example.features.feature['labels'].int64_list.value

        for label in labels:

            if label in video_count_dict:

                video_count_dict[label] = video_count_dict[label] + 1

                audio_sum_dict[label] = audio_sum_dict[label] + average_audio

                rgb_sum_dict[label] = rgb_sum_dict[label] + average_rgb

            else:

                video_count_dict[label] = 1

                audio_sum_dict[label] = 0

                rgb_sum_dict[label] = 0



audio_average_dict = {}

rgb_average_dict = {}



for key, elem in video_count_dict.items():

    audio_sum = audio_sum_dict[key]

    rgb_sum = rgb_sum_dict[key]

    

    audio_average_dict[key] = audio_sum/elem

    rgb_average_dict[key] =  rgb_sum/elem



sorted_rgb = sorted(rgb_average_dict.items(), key=operator.itemgetter(1))

i = 0

for value in sorted_rgb[::-1]:

    print(value[0])

    i = i + 1

    if i==20:

        break
filenames = ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]

for filename in filenames:

    print(filename)
feat_rgb = []

feat_audio = []

count = []



for example in tf.python_io.tf_record_iterator(frame_lvl_record):        

    tf_seq_example = tf.train.SequenceExample.FromString(example)

    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

    sess = tf.InteractiveSession()

    rgb_frame = []

    audio_frame = []

    # iterate through frames

    for i in range(n_frames):

        rgb_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        count.append(len(tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value))

        audio_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        

        

    sess.close()

    feat_rgb.append(rgb_frame)

    feat_audio.append(audio_frame)

    break
print('The first video has %d frames' %len(feat_rgb[0]))

print(rgb_frame[160])
print('Number of videos in this tfrecord: ',len(mean_rgb))

print('First video feature length',len(mean_rgb[0]))

print('First 20 features of the first youtube video (',vid_ids[0],')')

print(mean_rgb[0][:20])
def play_one_vid(record_name, video_index):

    return vid_ids[video_index]

    

# this worked on my local jupyter notebook, but doesn't show on kaggle kernels:

YouTubeVideo(play_one_vid(video_lvl_record, 7))
print('mean_audio has length of: ')

print([len(x) for x in mean_audio][:5])

print('mean_rgb has length of: ')

print([len(x) for x in mean_rgb][:5])
# now, let's read the frame-level data

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

                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        audio_frame.append(tf.cast(tf.decode_raw(

                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)

                       ,tf.float32).eval())

        

        

    sess.close()

    feat_rgb.append(rgb_frame)

    feat_audio.append(audio_frame)

    break
print('The first video has %d frames' %len(feat_rgb[0]))
from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.manifold import TSNE

import numpy as np
n=10

from collections import Counter

label_mapping = pd.Series.from_csv('../input/label_names.csv',header=0).to_dict()



top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)

top_n_labels = [int(i[0]) for i in top_n]

top_n_label_names = [label_mapping[x] for x in top_n_labels]

top_n_label_names
import networkx as nx

from itertools import combinations



G=nx.Graph()



G.clear()

for list_of_nodes in labels:

    filtered_nodes = set(list_of_nodes).intersection(set(top_n_labels))  

    for node1,node2 in list(combinations(filtered_nodes,2)): 

        node1_name = label_mapping[node1]

        node2_name = label_mapping[node2]

        G.add_node(node1_name)

        G.add_node(node2_name)

        G.add_edge(node1_name, node2_name)



nx.draw_networkx(G)
colors = plt.cm.rainbow(np.linspace(0, 1, n))

mean_rgb_top_n = []

labels_for_tsne = []

# filtering mean_rgb so it only contains top n labels

for idx, list_of_nodes in enumerate(labels):

    for node in list_of_nodes:

        if node in top_n_labels:

            mean_rgb_top_n.append(mean_rgb[idx])

            labels_for_tsne.append(node)





X_embedded = TSNE(n_components=2, random_state=0).fit_transform(mean_rgb_top_n) 





fig = plt.figure()

ax = fig.add_subplot(111)



handles = []

for indx, color in enumerate(colors):

    this_label = top_n_labels[indx]

    X_embedded_filtered = X_embedded[np.array([x==this_label for x in labels_for_tsne])]

    handles.append(ax.scatter(X_embedded_filtered[:, 0], X_embedded_filtered[:, 1], c=color, marker="o",edgecolor='none'))



ax.legend(handles, top_n_labels)



plt.show()