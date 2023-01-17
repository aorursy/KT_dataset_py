import tensorflow as tf
import pandas as pd
import numpy as np
print(tf.__version__)
# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('../input/taxi-train.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('../input/taxi-valid.csv', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('../input/taxi-test.csv', header = None, names = CSV_COLUMNS)
def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )
def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )
def make_prediction_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns
model = tf.estimator.LinearRegressor(feature_columns = make_feature_cols())
model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10))
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, df_valid)
predictions = model.predict(input_fn = make_prediction_input_fn(df_test))
for items in predictions:
  print(items)