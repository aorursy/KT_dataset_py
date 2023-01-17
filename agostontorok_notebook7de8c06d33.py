%pylab inline

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!ls ../input/video_level
import tensorflow as tf

import numpy as np

from IPython.display import YouTubeVideo





video_lvl_record = "../input/video_level/train-1.tfrecord"

frame_lvl_record = "../input/frame_level/train-1.tfrecord"
vid_ids = []

labels = []

mean_rgb = []

mean_audio = []



for example in tf.python_io.tf_record_iterator(video_lvl_record):

    tf_example = tf.train.Example.FromString(example)



    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(list(tf_example.features.feature['labels'].int64_list.value))

    mean_rgb.append(list(tf_example.features.feature['mean_rgb'].float_list.value))

    mean_audio.append(list(tf_example.features.feature['mean_audio'].float_list.value))
df = pd.DataFrame(mean_rgb)
df.mean().plot()


mat = df.corr()

np.fill_diagonal(mat.values, 0)



mat[mat.abs() < 0.1] = np.nan



plt.imshow(mat, figsize=(10,10))
mat.stack().abs().sort_values(ascending=False).head()
aa([item for item in tf_example.features.feature['mean_rgb'].float_list.value])
labels_df = pd.read_csv('../input/label_names.csv')

labels_df.head()
labels_df