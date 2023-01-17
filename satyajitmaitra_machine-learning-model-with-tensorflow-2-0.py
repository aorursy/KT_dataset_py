!pip freeze | grep tensorflow==2.3

import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep=",")
df.head()
df.describe()
np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]

def add_more_features(df):
  df['avg_rooms_per_house'] = df['total_rooms'] / df['households'] #expect positive correlation
  df['avg_persons_per_room'] = df['population'] / df['total_rooms'] #expect negative correlation
  return df
# Create pandas input function
def make_input_fn(df, num_epochs):
  return tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = add_more_features(df),
    y = df['median_house_value'] / 100000, 
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )
# Define your feature columns
def create_feature_cols():
  return [
    tf.feature_column.numeric_column('housing_median_age'),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), boundaries = np.arange(32.0, 42, 1).tolist()),
    tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'), boundaries = np.arange(-125.0, -113.0, 1).tolist()),
    tf.feature_column.numeric_column('avg_rooms_per_house'),
    tf.feature_column.numeric_column('avg_persons_per_room'),
    tf.feature_column.numeric_column('median_income')
  ]
# Create estimator train and evaluate function
def train_and_evaluate(output_dir,num_train_steps):
    estimator = tf.compat.v1.estimator.LinearRegressor(model_dir=output_dir,feature_columns=create_feature_cols())
    train_spec =tf.estimator.TrainSpec(input_fn=make_input_fn(traindf,None),
                                      max_steps=num_train_steps)
    
    eval_spec = tf.estimator.EvalSpec(input_fn=make_input_fn(evaldf,1),
                                     steps=None,
                                     start_delay_secs=1,
                                     throttle_secs=5)
    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
OUTDIR = "./trained_model"


# Run the model
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
tf.compat.v1.summary.FileWriterCache.clear() 
train_and_evaluate(OUTDIR, 2000)
!ls $OUTDIR
