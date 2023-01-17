# This Python 3 environment comes with many helpful analytics libraries instaled
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from kaggle_datasets import KaggleDatasets
GCS_PATH=KaggleDatasets().get_gcs_path("copymove")
print(GCS_PATH)

TFIDEN='CMFDTF'  # @param
IMG_DIM=256 
NB_CHANNEL=3 # @param
BATCH_SIZE=128 # @param
BUFFER_SIZE=1024 # @param
TRAIN_DATA=256*188 # @param
EVAL_DATA=256*25 # @param


GCS_PATH='{}/{}'.format(GCS_PATH,TFIDEN)
print(GCS_PATH)

import tensorflow as tf
import os 

def data_input_fn(mode): 
    
    def _parser(example):
        feature ={  'image'  : tf.io.FixedLenFeature([],tf.string) ,
                    'mask'   : tf.io.FixedLenFeature([],tf.string)
                    
        }    
        parsed_example=tf.io.parse_single_example(example,feature)
        image_raw=parsed_example['image']
        image=tf.image.decode_png(image_raw,channels=NB_CHANNEL)
        image=tf.cast(image,tf.float32)/255.0
        image=tf.reshape(image,(IMG_DIM,IMG_DIM,NB_CHANNEL))

        target_raw=parsed_example['mask']
        target=tf.image.decode_png(target_raw,channels=1)
        target=tf.cast(target,tf.float32)/255.0
        target=tf.reshape(target,(IMG_DIM,IMG_DIM,1))
        
        
        return image,target

    gcs_pattern=os.path.join(GCS_PATH,mode,'*.tfrecord')
    file_paths = tf.io.gfile.glob(gcs_pattern)
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parser)
    dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


import matplotlib.pyplot as plt
%matplotlib inline

eval_ds = data_input_fn("Eval")

for x,y in eval_ds.take(1):
    data=np.squeeze(x[0])
    plt.imshow(data)
    plt.show()
    print('Image Batch Shape:',x.shape)
    plt.imshow(np.squeeze(y[0]))
    plt.show()
    print('Target Batch Shape:',y.shape)
    print('Target Unique Values:',np.unique(np.squeeze(y[0])))
