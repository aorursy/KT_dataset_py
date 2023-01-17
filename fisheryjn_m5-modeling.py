# This Python 3 environment comes with many helpful analytics libraries installed

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
import re

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

from kaggle_datasets import KaggleDatasets
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

print("REPLICAS: ", strategy.num_replicas_in_sync)
!ls /kaggle/input/
GCS_PATH = KaggleDatasets().get_gcs_path('m5-accuracy-datasets')
train_path = GCS_PATH+'/train.csv'

validation_path = GCS_PATH+'/validation.csv'

test_path = GCS_PATH+'/test.csv'
def get_dataset(file_path):

    dataset = tf.data.experimental.make_csv_dataset(

        file_path,

        batch_size=1024, 

        label_name='value',

        num_epochs=1,

        ignore_errors=True)

    return dataset
train_data = get_dataset(train_path)

validation_data = get_dataset(validation_path)

test_data = get_dataset(test_path)
train_data = train_data.prefetch(buffer_size =100)

validation_data = validation_data.prefetch(buffer_size =100)
examples, labels = next(iter(train_data)) # 第一个批次

print("EXAMPLES: \n", examples, "\n")

print("LABELS: \n", labels)
numerical_columns = []



for feature in examples.keys():

    num_col = tf.feature_column.numeric_column(feature) #normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))

    numerical_columns.append(num_col)
preprocessing_layer = tf.keras.layers.DenseFeatures(numerical_columns)
with strategy.scope():

    model = tf.keras.Sequential([preprocessing_layer,

    tf.keras.layers.Dense(128, kernel_initializer = 'TruncatedNormal', activation = 'relu'),

    tf.keras.layers.Dense(256, kernel_initializer = 'TruncatedNormal', activation = 'relu'),                             

    tf.keras.layers.Dense(256,kernel_initializer = 'TruncatedNormal',activation = 'tanh'),

    tf.keras.layers.Dense(128,kernel_initializer = 'TruncatedNormal',activation = 'tanh'),

    tf.keras.layers.Dense(64,kernel_initializer = 'TruncatedNormal',activation = 'tanh'),

    tf.keras.layers.Dense(16,kernel_initializer = 'TruncatedNormal',activation = 'tanh'),

    tf.keras.layers.Dense(4,kernel_initializer = 'TruncatedNormal',activation = 'tanh'),                             

    tf.keras.layers.Dense(1)])# define your model normally

    model.compile(loss = 'mse', metrics=['mae'], optimizer = 'Adam')
history = model.fit(train_data,validation_data=validation_data, epochs = 5)
y_pred = model.predict(test_data)
y_pred.to_csv('y_pred.csv',index=False)
model.save('first_model.h5')