# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# The input for this competition is also available in BigQuery
# Using BigQuery will allow you to easily scale to large datasets
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# Any results you write to the current directory are saved as output.
query = """
SELECT * FROM (
    SELECT
        fare_amount,
        extract(DAYOFWEEK from pickup_datetime) as day_of_week,
        ABS(dropoff_longitude - pickup_longitude) as londiff,
        ABS(dropoff_latitude - pickup_latitude) as latdiff,
        passenger_count
    FROM
      `cloud-training-demos.taxifare_kaggle.train`
) 
WHERE 
  -- do some quality control
  londiff < 5.0 AND latdiff < 5.0 
  AND fare_amount > 1 AND fare_amount < 200
  -- sample the dataset for now. can remove the sampling later
  AND RAND() < 0.001
        """
bq_assistant = BigQueryHelper("cloud-training-demos", "taxifare_kaggle")
full_df = bq_assistant.query_to_pandas(query)
full_df.describe()
full_df = full_df.sample(frac=1.0) # shuffle
ntrain = (int)(len(full_df)*0.8)
eval_df = full_df[ntrain:]
train_df = full_df[:ntrain]
train_df.describe()
eval_df.describe()
train_df.head(n=3)
import tensorflow as tf
import tensorflow.feature_column as fc

# 3 input functions to feed data
# one each for train, eval and predict
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_df,
    y=train_df['fare_amount'],
    batch_size=128,
    num_epochs=None, # indefinitely
    shuffle=True
)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=eval_df, 
    y=eval_df['fare_amount'],
    num_epochs=1,
    shuffle=False
)
def serving_input_fn():
    ph = {
        'day_of_week' : tf.placeholder(tf.int32, [None]),
        'londiff': tf.placeholder(tf.float32, [None]),
        'latdiff': tf.placeholder(tf.float32, [None]),
        'passenger_count': tf.placeholder(tf.int32, [None])
    }
    features = ph # no transformations
    return tf.estimator.export.ServingInputReceiver(features, ph)

# train-and-evaluate loop. this code works distributed if submitted to Cloud ML Engine
def train_and_evaluate(outdir, num_train_steps):
    feature_cols = [
        fc.categorical_column_with_identity('day_of_week', num_buckets=8), # 0-7
        fc.numeric_column('londiff'),
        fc.numeric_column('latdiff'),
        fc.numeric_column('passenger_count'),
    ]
    estimator = tf.estimator.LinearRegressor(
        feature_columns=feature_cols, 
        model_dir=outdir
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn,
        max_steps = num_train_steps
    )
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_input_fn,
        start_delay_secs = 10, # start evaluating after N seconds
        throttle_secs = 60, # evaluate every N seconds
        exporters = exporter
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
import shutil, os
tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR='./taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 2000)
!ls taxi_trained
