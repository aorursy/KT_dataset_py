!pip install -q quick-ml
import tensorflow as tf

import quick_ml
import pandas as pd
df = pd.read_csv('../input/dog-breed-identification/labels.csv')
df['id'] = df['id'].apply(lambda x : x + '.jpg')
df
df.columns = ['Image', 'Id']
df
df.to_csv('train.csv', index = False)
train = pd.read_csv('train.csv')

train
from quick_ml.tfrecords_maker import create_split_tfrecords_from_csv



DATA_DIR = '../input/dog-breed-identification/train'

csv_path = './train.csv'

outfile1name = 'train.tfrecords'

outfile2name = 'val.tfrecords'

split_size_ratio = 0.7



create_split_tfrecords_from_csv(DATA_DIR,csv_path,  outfile1name, outfile2name, split_size_ratio, IMAGE_SIZE = (192,192))