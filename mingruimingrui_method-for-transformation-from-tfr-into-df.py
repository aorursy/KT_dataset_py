import numpy as np

import pandas as pd

import tensorflow as tf

from IPython.display import YouTubeVideo

from subprocess import check_output



labels = pd.read_csv('../input/label_names.csv') #4716 labels



cols_rgb = ["rgb_{}".format(i) for i in range(1024)]

cols_aud = ["aud_{}".format(i) for i in range(128)]

cols_y   = ["y_{}".format(i) for i in range(4716 + 1)]



usevidlvl = True

usefralvl = False
#PREPROCESSING



#homemade "dataframfy" method

def dffy(filenames):

    index = []

    rgb = []

    aud = []

    y = []

    if len(filenames)>0:

        isVideo = filenames[0].find("video")!=-1



    for filename in filenames:

        for example in tf.python_io.tf_record_iterator(filename):

            tf_example = tf.train.Example.FromString(example)

            index.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))

            if isVideo:

                rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)

                aud.append(tf_example.features.feature['mean_audio'].float_list.value)

            else:

                rgb.append(tf_example.features.feature['rgb'].float_list.value)

                aud.append(tf_example.features.feature['audio'].float_list.value)

            y.append(np.array(tf_example.features.feature['labels'].int64_list.value))

            

    index = pd.DataFrame(np.array(index),columns=["id"])

    rgb = pd.DataFrame(np.array(rgb),columns=cols_rgb)

    aud = pd.DataFrame(np.array(aud),columns=cols_aud)

    y2 = pd.DataFrame(np.zeros((len(index),len(labels) + 1)))

    for i in range(len(y)):

        y2.loc[i,y[i].clip(0,len(labels))] = 1

    y2.columns=cols_y

    y = []

    

    return [index,rgb,aud,y2]
if usevidlvl:

    vidfilenames = ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]

    [vid_id,vid_rgb,vid_aud,vid_y] = dffy(vidfilenames)



if usefralvl:

    frafilenames = ["../input/frame_level/train-{}.tfrecord".format(i) for i in range(10)]

    [fra_id,fra_rgb,fra_aud,fra_y] = dffy(frafilenames)