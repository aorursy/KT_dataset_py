import datetime

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



!pip install -q tensorflow==2.0.0-alpha0

import tensorflow as tf

# Load the TensorBoard notebook extension

%load_ext tensorboard.notebook



# Imports for the HParams plugin

from tensorboard.plugins.hparams import api_pb2

from tensorboard.plugins.hparams import summary as hparams_summary

from google.protobuf import struct_pb2



# Clear any logs from previous runs

!rm -rf ./logs/ 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
tf.__version__
data=pd.read_csv("../input/heart.csv")
data.head()
data.info()
data.describe()
# Just to see the correlation

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(method='pearson'),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
# A utility method to create a tf.data dataset from a Pandas Dataframe

def df_to_dataset(dataframe, shuffle=True, batch_size=32):

  dataframe = dataframe.copy()

  labels = dataframe.pop('target')

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds
feature_columns = []



# numeric cols

for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:

  feature_columns.append(tf.feature_column.numeric_column(header))



# bucketized cols

age = tf.feature_column.numeric_column("age")

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

feature_columns.append(age_buckets)



# indicator cols

data["thal"] = data["thal"].apply(lambda x: str(x))

thal = tf.feature_column.categorical_column_with_vocabulary_list(

      'thal', ['3', '6', '7'])

thal_one_hot = tf.feature_column.indicator_column(thal)

feature_columns.append(thal_one_hot)



data["sex"] = data["sex"].apply(str)

sex = tf.feature_column.categorical_column_with_vocabulary_list(

      'sex', ['0', '1'])

sex_one_hot = tf.feature_column.indicator_column(sex)

feature_columns.append(sex_one_hot)



data["cp"] = data["cp"].apply(lambda x: str(x))

cp = tf.feature_column.categorical_column_with_vocabulary_list(

      'cp', ['0', '1', '2', '3'])

cp_one_hot = tf.feature_column.indicator_column(cp)

feature_columns.append(cp_one_hot)



data["slope"] = data["slope"].apply(str)

slope = tf.feature_column.categorical_column_with_vocabulary_list(

      'slope', ['0', '1', '2'])

slope_one_hot = tf.feature_column.indicator_column(slope)

feature_columns.append(slope_one_hot)



# embedding cols

thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)

feature_columns.append(thal_embedding)



# crossed cols

age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)

age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)

feature_columns.append(age_thal_crossed)



cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)

cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)

feature_columns.append(cp_slope_crossed)
train, test = train_test_split(data, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')
batch_size = 32

train_ds = df_to_dataset(train, batch_size=batch_size)

val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
num_units_list = [128, 256]

dropout_rate_list = [0.2, 0.5] 

optimizer_list = ['adam', 'sgd'] 
# Utility method to create summary for tensorboard

def create_experiment_summary(num_units_list, dropout_rate_list, optimizer_list):

  num_units_list_val = struct_pb2.ListValue()

  num_units_list_val.extend(num_units_list)

  dropout_rate_list_val = struct_pb2.ListValue()

  dropout_rate_list_val.extend(dropout_rate_list)

  optimizer_list_val = struct_pb2.ListValue()

  optimizer_list_val.extend(optimizer_list)

  return hparams_summary.experiment_pb(

      # The hyperparameters being changed

      hparam_infos=[

          api_pb2.HParamInfo(name='num_units',

                             display_name='Number of units',

                             type=api_pb2.DATA_TYPE_FLOAT64,

                             domain_discrete=num_units_list_val),

          api_pb2.HParamInfo(name='dropout_rate',

                             display_name='Dropout rate',

                             type=api_pb2.DATA_TYPE_FLOAT64,

                             domain_discrete=dropout_rate_list_val),

          api_pb2.HParamInfo(name='optimizer',

                             display_name='Optimizer',

                             type=api_pb2.DATA_TYPE_STRING,

                             domain_discrete=optimizer_list_val)

      ],

      # The metrics being tracked

      metric_infos=[

          api_pb2.MetricInfo(

              name=api_pb2.MetricName(

                  tag='accuracy'),

              display_name='Accuracy'),

      ]

  )



exp_summary = create_experiment_summary(num_units_list, dropout_rate_list, optimizer_list)

root_logdir_writer = tf.summary.create_file_writer("logs/hparam_tuning")

with root_logdir_writer.as_default():

  tf.summary.import_event(tf.compat.v1.Event(summary=exp_summary).SerializeToString())
# Model compiler

def train_test_model(hparams):



  model = tf.keras.models.Sequential([

    tf.keras.layers.DenseFeatures(feature_columns),

    tf.keras.layers.Dense(hparams['num_units'], activation='relu'),

    tf.keras.layers.Dropout(hparams['dropout_rate']),

      tf.keras.layers.Dense(hparams['num_units'], activation='relu'),

    tf.keras.layers.Dense(2, activation='sigmoid')

  ])

  model.compile(optimizer=hparams['optimizer'],

                loss='binary_crossentropy',

                metrics=['accuracy'])



  model.fit(train_ds, 

            validation_data=val_ds, 

            epochs=50,

            use_multiprocessing=True,)

  _, accuracy = model.evaluate(test_ds)

  return accuracy
# Model runner

def run(run_dir, hparams):

  writer = tf.summary.create_file_writer(run_dir)

  summary_start = hparams_summary.session_start_pb(hparams=hparams)



  with writer.as_default():

    accuracy = train_test_model(hparams)

    summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)

      

    tf.summary.scalar('accuracy', accuracy, step=1, description="The accuracy")

    tf.summary.import_event(tf.compat.v1.Event(summary=summary_start).SerializeToString())

    tf.summary.import_event(tf.compat.v1.Event(summary=summary_end).SerializeToString())
session_num = 0



for num_units in num_units_list:

  for dropout_rate in dropout_rate_list:

    for optimizer in optimizer_list:

      hparams = {'num_units': num_units, 'dropout_rate': dropout_rate, 'optimizer': optimizer}

      print('--- Running training session %d' % (session_num + 1))

      print(hparams)

      run_name = "run-%d" % session_num

      run("logs/hparam_tuning/" + run_name, hparams)

      session_num += 1
# %tensorboard --logdir logs/hparam_tuning