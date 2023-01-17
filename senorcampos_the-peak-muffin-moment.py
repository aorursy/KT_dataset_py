



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import csv

import networkx as nx

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import tensorflow as tf

from IPython.display import YouTubeVideo

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os

#print(os.listdir("../input"))

#import warnings

#warnings.filterwarnings('ignore')
# first, let's use validation data instead of train data.

# only the validation data contains the segment start and times

# and segment labels



#frame_lvl_record = "../input/frame-sample/frame/train00.tfrecord"

frame_lvl_record = "../input/validate-sample/validate/validate00.tfrecord"
# there are two tensor flow records of each

print(os.listdir("../input/frame-sample/frame"))

print(os.listdir("../input/validate-sample/validate"))
vid_ids = []   # each video has an id

labels = []    # these labels refer to the entire video (a number 0 - 3861)

seg_start = [] # segment start times (appear to start only at multiples of 5 sec)

seg_end = []   # segment end times (end 5 seconds after the corresponding start)

seg_label = [] # the label for each start segment (same length as start, end)

seg_scores = [] # the score for that label.  Scores are 1 = present, 0 = not present

for example in tf.python_io.tf_record_iterator(frame_lvl_record):

    tf_example = tf.train.Example.FromString(example)

    # thanks to the original kernel authors for figuring out how to do this:

    vid_ids.append(tf_example.features.feature['id']

                   .bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)

    seg_start.append(tf_example.features.feature['segment_start_times'].int64_list.value)

    seg_end.append(tf_example.features.feature['segment_end_times'].int64_list.value)

    seg_label.append(tf_example.features.feature['segment_labels'].int64_list.value)

    seg_scores.append(tf_example.features.feature['segment_scores'].float_list.value)

# my technique:

# manually go through a few videos until I find a video for which 

# the seg_scores are not uniformly 1.0

# in rec_id = 1, all of the seq_scores are 1.0, so this video will not 

# give much insight as to how a video could have precise event start time

# rec_id = 5 has 0, 1, 1, 1, 0, which looks more promising

rec_id = 5

print(labels[rec_id])

print(seg_start[rec_id])

print(seg_label[rec_id])

print(seg_scores[rec_id])



print('Picking a youtube video id:',vid_ids[rec_id])
# quick verification of the segment label

vocabulary = pd.read_csv('../input/vocabulary.csv')

vocabulary[vocabulary['Index']==1672]
# With that video id, we can play the video

# apparently if you take the videoID, let's call it vvDD, 

# then you can put the following in a browser:

# http://data.yt8m.org/2/j/i/vv/vvDD.js

# so for rec_id = 5 of the first validation example, vvDD = Kt00,

# so pointing a browser here: http://data.yt8m.org/2/j/i/Kt/Kt00.js

# yields this: i("Kt00","MZYaCFJogqo");

# I'd be interested in a more pythonic way of doing that.
# let's watch the video:



#YouTubeVideo('FBQ00Vk7Obs')  #op00

#YouTubeVideo('1Cb84yXZgZs')   #O900

YouTubeVideo('MZYaCFJogqo')  #Kt00
# let's read the video features (feat_rgb), 

# and the audio features while we're at it 

# (thanks again to earlier kernel providers)



feat_rgb = []

feat_audio = []



# the example that I started with read the first record and then did a break

# the continue statements at the beginning get us to the record of interest

# but then do the same thing - break after reading in the video feature data.

cur_rec = -1

for example in tf.python_io.tf_record_iterator(frame_lvl_record):

    cur_rec += 1

    if cur_rec < rec_id:

        continue

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

print('The video has %d frames' %len(feat_rgb[0]))
# from here on out we'll switch to exploration in numpy

# which I am more familiar with

# first, we'll convert the rgb_frame values into a numpy array

rgb_frame = np.array(rgb_frame)

# the numpy array has the shape: (num frames, 1024)

# where 1024 is the number of abstract "features" output by a neural network

rgb_frame.shape
# let's take a look at these abstracted features:

# the x-axis is time in seconds, the y-axis is the list of features

plt.imshow(rgb_frame.T, aspect='auto');
# the following is a dot product of the feature vector with itself

# it will yield a matrix of the size (num seconds, num seconds)

# the value of the matrix will correspond to self-similarity of different seconds  

cross_t = np.dot(rgb_frame,rgb_frame.T)

cross_t.shape
# let's look at it

plt.imshow(cross_t, aspect = 'auto');