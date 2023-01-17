import os

import skimage.io

from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import numpy as np

import shutil

import cv2

import math
import matplotlib.pyplot as plt



def xy_visualization(x, y = None):

    plt.imshow(x)

    plt.axis('off')

    plt.show()

    if y is not None:

        print(y)
df = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv') 

df['bbox'] = df['bbox'].apply(lambda x: x[1:-1].split(",")) 

df['x'] = df['bbox'].apply(lambda x: x[0]).astype('float32').astype('int32') 

df['y'] = df['bbox'].apply(lambda x: x[1]).astype('float32').astype('int32') 

df['w'] = df['bbox'].apply(lambda x: x[2]).astype('float32').astype('int32') 

df['h'] = df['bbox'].apply(lambda x: x[3]).astype('float32').astype('int32') 

df = df[['image_id','x', 'y', 'w', 'h']]



cache = {}

for indx, row in tqdm(df.iterrows()):

    image_id, x, y, w, h = row['image_id'], row['x'], row['y'], row['w'], row['h']

    x1, y1, x2, y2 = x, y, x + w, y + h

    if image_id not in cache:

        cache[image_id] = {

            "x" : '../input/global-wheat-detection/train/' + image_id + '.jpg',

            "y": np.zeros((120, 5), np.float32),

            'y_cnt': 0

        }

        cache[image_id]["y"][:, 4] = -1 # Set all classes_idx to -1 

    cache[image_id]["y"][cache[image_id]["y_cnt"]] = [x1, y1, x2, y2, 0]

    cache[image_id]["y_cnt"] += 1

    

cache = [i for i in cache.values()]

# cv2.cvtColor(cv2.imread('../input/global-wheat-detection/train/' + image_id + '.jpg'), cv2.COLOR_BGR2RGB),
print((len(cache)))
import numpy as np

import tensorflow as tf

import os

import random

from tqdm import tqdm



class encode_and_write:

    def __init__(self):

        self.feature_dict = {

            'ndarray' : self._ndarray_feature, 

            'bytes' : self._bytes_feature, 

            'float' : self._float_feature,

            'double' : self._float_feature, 

            'bool' : self._int64_feature,

            'enum' : self._int64_feature, 

            'int' : self._int64_feature,

            'uint8' : self._int64_feature,

            'int32' : self._int64_feature,

            'uint' : self._int64_feature

        }

    def _ndarray_feature(self, value):

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

    

    def _bytes_feature(self, value):

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



    def _float_feature(self, value):

        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



    def _int64_feature(self, value):

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    

    def _encode_example(self, example):

        """Creates a tf.Example message ready to be written to a file."""

        feature = {}

        for vname in example:

            vtype = type(example[vname]).__name__

            feature[vname] = self.feature_dict[vtype](example[vname])

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return example_proto.SerializeToString()



    def run(self, filename, datasets, split_thresold=1000):

        datasets_itor = iter(datasets)

        

        try:

            for file_idx in range(100000000):

                pre_read = self._encode_example(datasets_itor.__next__())

                with tf.io.TFRecordWriter('%s_%d.tfrecord'%(filename, file_idx)) as writer:

                    for k in tqdm(range(split_thresold)):

                        if pre_read is not None:

                            writer.write(pre_read)

                            pre_read = None

                        else:

                            writer.write(self._encode_example(datasets_itor.__next__()))

        except StopIteration:

            pass

                    

  

class datasets_stream:

    def __init__(self, group_item_cnt, group_idx):

        self.group_item_cnt = group_item_cnt

        self.group_idx = group_idx

        

    def __iter__(self):

        np.random.seed(0)

        self.imgs_idxs = np.random.permutation([i for i in range(len(cache))])[self.group_item_cnt * self.group_idx: self.group_item_cnt * (self.group_idx + 1)]

        self.imgs_idx = 0

        return self



    def __next__(self):

        while True:

            if self.imgs_idx < len(self.imgs_idxs):

                idx = self.imgs_idxs[self.imgs_idx]

                self.imgs_idx += 1

                x, y = cv2.cvtColor(cv2.imread(cache[idx]["x"]), cv2.COLOR_BGR2RGB), cache[idx]["y"]

                return {"x": x, "y": y }

            else:

                raise StopIteration



            

encode_and_write().run('./train', datasets_stream(675, 0), split_thresold=64)