

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import clear_output



from tensorflow import keras

from tensorflow.keras import layers



tf.keras.backend.set_floatx('float64')



batch_size=50

max_steps=1000

df = pd.read_csv("/kaggle/input/bike-rental-prediction/train.csv")

df
def prepare_data(dataframe):

  dataframe['season'] = dataframe['season'].map({ 1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})

  dataframe['weathersit'] = dataframe['weathersit'].map({1: 'clear', 2: 'cloudy', 3: 'light_rain', 4: 'heavy_rain'})



  dataframe['temp'] = dataframe['temp'] / 3

  dataframe['atemp'] = dataframe['atemp'] / 3

  dataframe['hum'] = dataframe['hum'] / 10

  dataframe['windspeed'] = dataframe['windspeed'] / 5



  dataframe['temp'] = dataframe['temp'].astype(int)

  dataframe['atemp'] = dataframe['atemp'].astype(int)

  dataframe['hum'] = dataframe['hum'].astype(int)

  dataframe['windspeed'] = dataframe['windspeed'].astype(int)





  dataframe = dataframe.drop(['instant', 'dteday', 'year', 'registered', 'casual',], axis=1, errors='ignore')

  dataframe = pd.get_dummies(dataframe, prefix='', prefix_sep='')

  

  return dataframe

df = prepare_data(df)



df
train_df = df.sample(frac=0.95,random_state=0)

test_df = df.drop(train_df.index)



train_df
sns.pairplot(train_df[['cnt', 'temp', 'hum', 'windspeed']], diag_kind="kde")
train_labels = train_df.pop('cnt')

test_labels = test_df.pop('cnt')
NUMERIC_COLUMNS = ['temp', 'atemp', 'hum', 'windspeed']



COLUMNS = train_df.columns;

CATEGORICAL_COLUMNS =  []



for col in train_df.columns:

  if col not in NUMERIC_COLUMNS:

    CATEGORICAL_COLUMNS.append(col)



feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:

  vocabulary = train_df[feature_name].unique()

  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))



for feature_name in NUMERIC_COLUMNS:

  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))



temp_x_hum = tf.feature_column.crossed_column(['temp', 'hum'], hash_bucket_size=7000)

temp_x_wind = tf.feature_column.crossed_column(['temp', 'windspeed'], hash_bucket_size=7000)

hr_x_weekday = tf.feature_column.crossed_column(['hr', 'weekday'], hash_bucket_size=100)



feature_columns.append(temp_x_hum)

feature_columns.append(hr_x_weekday)

feature_columns.append(temp_x_wind)
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=train_df , y=train_labels, batch_size=batch_size, num_epochs=None, shuffle=True)

eval_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=test_df , y=test_labels, num_epochs=1, shuffle=False)

model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

model.train(input_fn=train_input_fn, max_steps=max_steps)
result = model.evaluate(eval_input_fn)



clear_output()

print(result)
dftest_orginal = pd.read_csv("/kaggle/input/bike-rental-prediction/test.csv")

dftest = dftest_orginal.copy()

dftest = prepare_data(dftest)



test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=dftest, y=None, batch_size=1, shuffle=False)

test_predictions = list(model.predict(test_input_fn))



from datetime import datetime

now = datetime.now()

timestamp = datetime.timestamp(now)



dftest_orginal['cnt'] = [int(round(t['predictions'][0])) if t['predictions'][0] > 0 else 0 

                         for t in test_predictions]

prefix = f'{timestamp}_{max_steps}_{batch_size}'



dftest_orginal.to_csv(prefix + 'full-result.csv')



columns = ['instant', 'cnt']

dftest_orginal.to_csv(f'{prefix}_bartek-result.csv', columns=columns, index=False)