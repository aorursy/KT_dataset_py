# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '../input/arabic-sentiment-twitter-corpus/arabic_tweets'
! pip install tf-nightly 
import tensorflow as tf 

tf.__version__
batch_size = 32

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir,

    batch_size=batch_size,

    validation_split=0.2,

    subset="training",

    seed=1337,

)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir,

    batch_size=batch_size,

    validation_split=0.2,

    subset="validation",

    seed=1337,

)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(

    data_dir, batch_size=batch_size

)



print(

    "Number of batches in raw_train_ds: %d"

    % tf.data.experimental.cardinality(raw_train_ds)

)

print(

    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)

)

print(

    "Number of batches in raw_test_ds: %d"

    % tf.data.experimental.cardinality(raw_test_ds)

)
for text_batch, label_batch in raw_train_ds.take(1):

    for i in range(5):

        print(text_batch.numpy()[i].decode('utf-8').strip())

        print(label_batch.numpy()[i])

        print('--------------------------------')