
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

print(tf.__version__)
# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['X', 'Y']
FEATURES = CSV_COLUMNS[0:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('../input/dataTraining.txt', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('../input/dataTraining.txt', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('../input/dataTraining.txt', header = None, names = CSV_COLUMNS)
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
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
tf.logging.set_verbosity(tf.logging.ERROR)

OUTDIR = '../input/trained_model'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols())#, model_dir = OUTDIR)  because it is Read-only file system

model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10))


#model = tf.estimator.DNNRegressor(hidden_units = [32, 8, 2],
#      feature_columns = make_feature_cols(), model_dir = OUTDIR)
#model.train(input_fn = make_train_input_fn(df_train, num_epochs = 100))

print_rmse(model, df_valid)
predictions = model.predict(input_fn = make_prediction_input_fn(df_test))
for items in predictions:
  print(items)