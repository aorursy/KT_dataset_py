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



video_lvl_record = "../input/video_level/train-1.tfrecord"

frame_lvl_record = "../input/frame_level/train-1.tfrecord"



vid_ids = []

labels = []

mean_rgb = []

mean_audio = []



for example in tf.python_io.tf_record_iterator(video_lvl_record):

    tf_example = tf.train.Example.FromString(example)



    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)

    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)

    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
n=20

from collections import Counter

label_mapping = pd.Series.from_csv('../input/label_names.csv',header=0)

label_dict = label_mapping.to_dict()



top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)

top_n_labels = [int(i[0]) for i in top_n]

top_n_label_count = [int(i[1]) for i in top_n]

top_n_label_names = [label_dict[x] for x in top_n_labels]



top_labels = pd.DataFrame(data = top_n_labels, columns = ['label_num'])

top_labels['count'] = top_n_label_count

top_labels['label_name'] = top_n_label_names



top_labels = top_labels.drop('label_num', axis = 1)



import seaborn as sns

import matplotlib.pyplot as plt



ax = sns.barplot(x='label_name', y='count', data=top_labels)

ax.set(xlabel='Label Name', ylabel='Label Count')

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

_ = plt.show();
#### Next step:

#### Find a way of taking the lists generated above to be data frames

#### Do more EDA on that
### Trying to figure out how to get a list of videos in the top 20..



test = labels[:5]

label_test = [item for sublist in test for item in sublist]

label_test