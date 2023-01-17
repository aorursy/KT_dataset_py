import tensorflow as tf

import numpy as np

from IPython.display import YouTubeVideo
vid_ids = []

labels = []

seg_start = []

seg_end = []

seg_label = []

seg_scores = []

validate_record = "../input/validate-sample/validate/validate00.tfrecord"

for example in tf.python_io.tf_record_iterator(validate_record):

    tf_example = tf.train.Example.FromString(example)

    vid_ids.append(tf_example.features.feature['id']

                   .bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)

    seg_start.append(tf_example.features.feature['segment_start_times'].int64_list.value)

    seg_end.append(tf_example.features.feature['segment_end_times'].int64_list.value)

    seg_label.append(tf_example.features.feature['segment_labels'].int64_list.value)

    seg_scores.append(tf_example.features.feature['segment_scores'].float_list.value)
import pandas as pd

vocab = pd.read_csv('../input/vocabulary.csv')

label_mapping =  vocab[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']
print('The first video id:',vid_ids[0])

print('Label of this video:',labels[0])

print('Segment start of this video:',seg_start[0])

print('Segment label of this video:',seg_label[0])

print('Segment Score of this video:',seg_scores[0])

print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[0]))])
YouTubeVideo('DxWJGOZL1co')
print('The 2nd video id:',vid_ids[1])

print('Label of this video:',labels[1])

print('Segment start of this video:',seg_start[1])

print('Segment label of this video:',seg_label[1])

print('Segment Score of this video:',seg_scores[1])

print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[1]))])
YouTubeVideo('JdYkqQFprUI')
print('The next video id:',vid_ids[2])

print('Label of this video:',labels[2])

print('Segment start of this video:',seg_start[2])

print('Segment label of this video:',seg_label[2])

print('Segment Score of this video:',seg_scores[2])

print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[2]))])
YouTubeVideo('f9wADEgGuH8')
print('The next video id:',vid_ids[3])

print('Label of this video:',labels[3])

print('Segment start of this video:',seg_start[3])

print('Segment label of this video:',seg_label[3])

print('Segment Score of this video:',seg_scores[3])

print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[3]))])
YouTubeVideo('4C8kuTvHXqQ')