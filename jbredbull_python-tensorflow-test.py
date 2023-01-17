import numpy as np 
import pandas as pd 
import os
train_dataset_full = pd.read_csv('../input/08_train_dataset.csv')
test_dataset = pd.read_csv('../input/08_test_dataset_no_response.csv')
train_dataset_full.head()
train_dataset_full.describe()
train_dataset_full['price_level'].unique()
train_dataset_full['rating'].unique()
train_dataset_full.info()
import seaborn as sns
ax = sns.regplot(x = "vpo_stores_nearby", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "closest_store_vpo", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "closest_store_dist", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "stores_count_nearby", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "population_2018_year_average_total_number", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "population_2018_year_average_per_mill_of_country", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "purchasing_power_2018_million_euro", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "purchasing_power_2018_per_mill_of_country", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "purchasing_power_2018_euro_per_capita", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
ax = sns.regplot(x = "purchasing_power_2018_index_country_eq_100", y = "volume_in_cans", ci = None, truncate = False, data = train_dataset_full)
import tensorflow as tf
LABEL='volume_in_cans'
#FEATURE_COLUMNS = list(train_dataset_full.columns)
#FEATURE_COLUMNS.remove(LABEL)
#FEATURE_COLUMNS = ['vpo_stores_nearby']
FEATURE_COLUMNS = ['vpo_stores_nearby','closest_store_vpo','closest_store_dist','stores_count_nearby',
                   'population_2018_year_average_total_number','population_2018_year_average_per_mill_of_country',
                  'purchasing_power_2018_million_euro','purchasing_power_2018_per_mill_of_country','purchasing_power_2018_euro_per_capita',
                  'purchasing_power_2018_index_country_eq_100']
def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df[FEATURE_COLUMNS],
    y = df[LABEL],
    batch_size = 256,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )
def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df[FEATURE_COLUMNS],
    y = df[LABEL],
    batch_size = 256,
    shuffle = False,
    queue_capacity = 1000
  )
def make_prediction_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df[FEATURE_COLUMNS],
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURE_COLUMNS]
  return input_columns
import shutil
tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = './test_model'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols(), model_dir = OUTDIR)

model.train(input_fn = make_train_input_fn(train_dataset_full, num_epochs = 1))
def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, train_dataset_full)
predictions = model.predict(input_fn = make_prediction_input_fn(test_dataset))
predicted_vol = [item['predictions'][0] for item in predictions]
test_dataset['lr_test_preds'] = predicted_vol
from time import time
test_dataset[['google_place_id','lr_test_preds']].to_csv(f"submission-{int(time())}.csv", index=False)
