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





video_lvl_record = "../input/video_level/train-5.tfrecord"

frame_lvl_record = "../input/frame_level/train-5.tfrecord"
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