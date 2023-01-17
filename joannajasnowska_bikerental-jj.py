import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import math as mt

train = pd.read_csv('train.csv');
test = pd.read_csv('test.csv');  

continuous = ['season', 'hr', 'workingday', 'weathersit', 'atemp', 'hum']
scaler = StandardScaler()
for var in continuous:
  train[var] = train[var].astype('float64')
  train[var] = scaler.fit_transform(train[var].values.reshape(-1, 1))
  test[var] = test[var].astype('float64')
  test[var] = scaler.fit_transform(test[var].values.reshape(-1, 1))

train.drop(['dteday','casual', 'registered', 'instant', 'temp', 'holiday','year', 'mnth', 'windspeed', 'weekday' ], axis=1, inplace=True, errors='ignore');  
test.drop(['dteday','casual', 'registered', 'instant', 'temp', 'holiday','year', 'mnth', 'windspeed', 'weekday' ], axis=1, inplace=True, errors='ignore'); 

x_df = train.drop(['cnt'], axis=1)
y_df = train['cnt']

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=1)

feat_cols = ['season', 'hr', 'workingday', 'weathersit', 'atemp']
feature_cols = [tf.compat.v1.feature_column.numeric_column(k) for k in feat_cols]			

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

model = tf.estimator.DNNRegressor(
    feature_columns=feature_cols,
    hidden_units=[1024, 512, 256],
    optimizer=lambda: tf.keras.optimizers.Adam(
        learning_rate=tf.compat.v1.train.exponential_decay(
            learning_rate=0.1,
            global_step=tf.compat.v1.train.get_global_step(),
            decay_steps=10000,
            decay_rate=0.96)))

model.train(input_fn=input_func, max_steps=10000)

model_eval_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=len(x_train), shuffle=False)
model_eval_predictions = list(model.predict(input_fn=model_eval_func))
model_eval_final_preds = []
for pred in list(model_eval_predictions):
  model_eval_final_preds.append(pred['predictions'][0] if pred['predictions'][0] > 0 else 0)

from sklearn.metrics import mean_squared_log_error
print('result : ')
print( mean_squared_log_error(y_train, model_eval_final_preds))


test_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=test2, batch_size=len(test), shuffle=False, target_column='cnt')
predictions = list(model.predict(input_fn=test_func))
final_preds = []
for pred in list(predictions):
  final_preds.append(pred['predictions'][0] if pred['predictions'][0] > 0 else 0)

filename = 'Bike predictions.csv'
submission.to_csv(filename, index = False)
