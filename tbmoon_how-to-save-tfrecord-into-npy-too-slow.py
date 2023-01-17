# Reference: https://www.kaggle.com/inversion/starter-kernel-yt8m-2019-sample-data (Thanks!)

# How to save tfrecord into numpy for pytorch user (TOO SLOW!)

# Please remove "break" in In[4] to experience how it is really slow.

# Do you have any ideas? 



import os

import numpy as np

import tensorflow as tf
frame_dir = '../input/validate-sample/validate/'

out_dir = '../out/'

try:

    os.stat(out_dir)

except:

    os.mkdir(out_dir)       
frame_lvl_record = frame_dir + 'validate00.tfrecord'
for example in tf.python_io.tf_record_iterator(frame_lvl_record):



    dataset = dict()

    

    tf_example = tf.train.Example.FromString(example)

    tf_seq_example = tf.train.SequenceExample.FromString(example)



    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)



    vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')

    vid_labels = tf_example.features.feature['labels'].int64_list.value    



    rgb_frame = []

    audio_frame = []

    

    sess = tf.InteractiveSession()



    for i in range(n_frames):

        rgb_frame.append(tf.cast(tf.decode_raw(

            tf_seq_example.feature_lists.feature_list['rgb']

            .feature[i].bytes_list.value[0], tf.uint8)

                                 , tf.float32).eval())

        

        audio_frame.append(tf.cast(tf.decode_raw(

            tf_seq_example.feature_lists.feature_list['audio']

            .feature[i].bytes_list.value[0], tf.uint8)

                                   , tf.float32).eval())



    sess.close()

    

    dataset['id'] = vid_id

    dataset['labels'] = list(vid_labels)

    dataset['rgb_frame'] = list(rgb_frame)

    dataset['audio_frame'] = list(audio_frame)

            

    np.save(out_dir + vid_id + '.npy', np.array(dataset))

    

    break  # read only the 1st video
print(os.listdir(out_dir))