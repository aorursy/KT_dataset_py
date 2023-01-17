import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

import xgboost as xgb
# shapes of raw data

yr_2016 = pd.read_csv('../input/pred2016l.csv')

yr_2017 = pd.read_csv('../input/pred2017l.csv')

print('Year 2016:', yr_2016.shape)

print('Year 2017:', yr_2017.shape)
# train-valid-test split

X_temp = yr_2016.drop('Ticks',axis=1)

y_temp = yr_2016['Ticks']



X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, train_size=0.75)



X_test  = yr_2017.drop('Ticks',axis=1)

y_test  = yr_2017['Ticks']



print(X_train.shape)

print(y_train.shape)

print(X_valid.shape)

print(y_valid.shape)

print(X_test.shape)
# Convert our data into XGBoost format

d_train = xgb.DMatrix(X_train, y_train)

d_valid = xgb.DMatrix(X_valid, y_valid)

d_test  = xgb.DMatrix(X_test)
#

xgb_params = {

    'n_trees': 75, 

    'eta': 0.03,

    'max_depth': 4,

    'subsample': 0.90,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    #'base_score': y_mean, # base prediction = mean(target)

    'silent': 0}



#

num_boost_rounds = 1000

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# Train the model!

mdl = xgb.train(dict(xgb_params, silent=0), d_train, num_boost_rounds, watchlist, early_stopping_rounds=500, maximize=False, verbose_eval=10)
#

y_pred = mdl.predict(d_test)
print(y_pred.shape)

print(y_test.shape)