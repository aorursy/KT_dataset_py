# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data_df=pd.read_csv('../input/global-wheat-detection/train.csv')

train_data_df.head()
image_id=[f'{i}.jpg' for i in train_data_df.image_id]

xmins,ymins,xmaxs,ymaxs,area=[],[],[],[],[]

for bbox in train_data_df.bbox:

    real_bbox=eval(bbox)

    

    xmin, ymin ,w ,h=real_bbox

    

    

    

    a=int(xmin+w)

    b=int(ymin+h)

    xmaxs.append(a)

    ymaxs.append(b)



    

    c=int(xmin)

    d=int(ymin)

    xmins.append(c)

    ymins.append(d)

    

    area.append(w*h)

    

data=pd.DataFrame()

data['filename']=image_id

data['width']=train_data_df.width

data['width']=train_data_df.height



data['class']=['wheat']*len(image_id)



data['xmin']=xmins

data['ymin']=ymins



data['xmax']=xmaxs

data['ymax']=ymaxs



data['iscrowd']=[0]*len(image_id)



data['area']=area

data['source']=train_data_df.source



data
# https://www.kaggle.com/raininbox/check-clean-big-small-bboxes

data=data.drop(data[(data["area"]<300) | (data["xmax"]-data["xmin"]<10) | (data["xmax"]-data["xmin"]<10)].index)

data=data.drop([173,2169,118211,52868,117344,3687,2159,121633,113947])

data.reset_index(drop=True, inplace=True)

data
from collections import namedtuple



width = 1024

height = 1024

columns = list(data)

grouped_data = []



def split(df, group):

    gb = df.groupby(group)

    return [namedtuple('data', ['filename', 'object'])(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]





groups = split(data, 'filename')



for group in groups:

    xmins = []

    xmaxs = []

    ymins = []

    ymaxs = []

    classes_text = []

    classes = []

    iscrowd=[]

    area=[]

    for index, row in group.object.iterrows():

        xmins.append(row['xmin'] / width)

        xmaxs.append(row['xmax'] / width)

        ymins.append(row['ymin'] / height)

        ymaxs.append(row['ymax'] / height)

        iscrowd.append(row["iscrowd"])

        area.append(row["area"])

        classes_text.append(row['class'].encode('utf8'))

        classes.append(1)

    grouped_data.append(dict(zip(columns + ['class_text'], [row['filename'], width, classes, xmins, ymins, xmaxs, ymaxs, iscrowd, area, row['source'], classes_text])))



dataset = pd.DataFrame(grouped_data)

dataset
from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

dataset.loc[:, 'fold'] = 0

for fold_number , (train_index, val_index) in enumerate(skf.split(dataset.index.values, y=dataset['source'].values)):

    dataset.loc[val_index, 'fold'] = fold_number

dataset = dataset.sort_values(['fold'])
# https://www.kaggle.com/alexandersoare/how-to-prepare-a-stratified-split/comments

import matplotlib.pyplot as plt



train_df = dataset[dataset['fold'] != 0]

val_df = dataset[dataset['fold'] == 0]



fig = plt.figure(figsize=(20, 5))

counts = train_df['source'].value_counts()

ax1 = fig.add_subplot(1,2,1)

a = ax1.bar(counts.index, counts)

counts = val_df['source'].value_counts()

ax2 = fig.add_subplot(1,2,2)

a = ax2.bar(counts.index, counts)
import tensorflow as tf

from PIL import Image

import hashlib

import io



def int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))





def int64_list_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))





def bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def bytes_list_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))





def float_list_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def create_tf_example(item, i):

    with tf.io.gfile.GFile(os.path.join('../input/global-wheat-detection/train', '{}'.format(item.filename)), 'rb') as fid:

        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)

    key = hashlib.sha256(encoded_jpg).hexdigest()

    filename = item.filename.encode('utf8')

    width = 1024

    height = 1024



    tf_example = tf.train.Example(features=tf.train.Features(feature={

        'image/height': int64_feature(height),

        'image/width': int64_feature(width),

        'image/filename': bytes_feature(filename),

        'image/source_id':bytes_feature(str(i).encode('utf8')),

        'image/key/sha256':bytes_feature(key.encode('utf8')),

        'image/encoded':bytes_feature(encoded_jpg),

        'image/format': bytes_feature('jpg'.encode('utf8')),

        'image/object/bbox/xmin': float_list_feature(item.xmin),

        'image/object/bbox/xmax': float_list_feature(item.xmax),

        'image/object/bbox/ymin': float_list_feature(item.ymin),

        'image/object/bbox/ymax': float_list_feature(item.ymax),

        'image/object/class/text':bytes_list_feature(item.class_text),

        'image/object/class/label':int64_list_feature(item['class']),

        'image/object/is_crowd':int64_list_feature(item.iscrowd),

        'image/object/area':float_list_feature(item.area)

    }))

    return tf_example



for fold in range(0,10):

    val_df = dataset[dataset['fold'] == fold]

    train_writer = tf.io.TFRecordWriter(f'{fold}.tfrecord')

    for i, row in val_df.iterrows():    

        tf_example = create_tf_example(row, i)       

        train_writer.write(tf_example.SerializeToString())