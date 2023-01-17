# Imports

import numpy as np

import pandas as pd

import os



from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge, RidgeCV



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

pd.options.display.precision = 15



import psutil

import gc

from catboost import CatBoostRegressor

import seaborn as sns

from scipy import stats



import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook
# Read the csv file to take input into train dataframe using pandas, 

# 2 attributes from the training data: acoustic data: records the seismic activity and

# time_to_failure: time left for the next laboratory earthquake

train_dataset = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
# number of instances for each segment to be specified as 150000 because each test segment has 150000 observations.

instances = 150000

segments = int(np.floor(train_dataset.shape[0] / instances))



# X_trainset is the training data: acoustic data

# y_trainset is the value to be predicted, that is the time left for the next lab earthquake

X_trainset = pd.DataFrame(index=range(segments), dtype=np.float64)

y_trainset = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



# for every segment id, do feature engineering

for segment in tqdm_notebook(range(segments)):



#   creating segments of size 150000 starting from the row of segment id and upto segment id + instances

    each_seg = train_dataset.iloc[segment * instances : segment * instances + instances]

#     create x and y having acoustic_data and time_to _failure respectively

    x_rawdata = each_seg['acoustic_data']

    x = x_rawdata.values

    y = each_seg['time_to_failure'].values[-1]

    

#     y train data is the time_to_failure for that segment instance

    y_trainset.loc[segment, 'time_to_failure'] = y

    X_trainset.loc[segment, 'average'] = x.mean()   # average of all acoustic data values for segment instance

    X_trainset.loc[segment, 'standard_deviation'] = x.std()    # standard deviation

    X_trainset.loc[segment, 'maximum'] = x.max()    # maximum value

    X_trainset.loc[segment, 'minimum'] = x.min()    # minimum value

    X_trainset.loc[segment, 'quantile_1_percentile'] = np.quantile(x,0.01)    # the value below which 1% of data appears in the acoustic_data attribute

    X_trainset.loc[segment, 'quantile_5_percentile'] = np.quantile(x,0.05)    # the value below which 5%

    X_trainset.loc[segment, 'quantile_95_percentile'] = np.quantile(x,0.95)    # the value below which 95%

    X_trainset.loc[segment, 'quantile_99_percentile'] = np.quantile(x,0.99)    # the value below which 99%

    X_trainset.loc[segment, 'median_absolute'] = np.median(np.abs(x))        # median of absolute values of acoustic_data

    X_trainset.loc[segment, 'quantile_95_percentile_absolute'] = np.quantile(np.abs(x),0.95)    # the absolute value below which 95% of absolute acoustic_data data 

    X_trainset.loc[segment, 'quantile_99_percentile_absolute'] = np.quantile(np.abs(x),0.99)    # the absolute value below which 99% of absolute acoustic_data data 

    

#     divide the data into group of 5; each of size 30000 and perform ANOVA tests to check

#     if thesse groups have same population mean hence helping us determine the variance of the data.

    X_trainset.loc[segment, 'F_test_measure'], X_trainset.loc[segment, 'p_test_measure'] = stats.f_oneway(x[:30000],x[30000:60000],x[60000:90000],x[90000:120000],x[120000:])



#     .diff will give the change in x with respect to it's previous value; mean of all such changes.

    X_trainset.loc[segment, 'average_change_absolute'] = np.mean(np.diff(x))      

    

#     take change values and divide by itself, then consider only those which come out to be non-zero

    X_trainset.loc[segment, 'average_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_trainset.loc[segment, 'maximum_absolute'] = np.abs(x).max()     # max of absolute values in acoustic_data

    

    for windows in [10,100]:

        x_roll_std = x_rawdata.rolling(windows).std().dropna().values

        x_roll_mean = x_rawdata.rolling(windows).mean().dropna().values

        

        X_trainset.loc[segment, 'average_rolling_standard_deviation' + str(windows)] = x_roll_std.mean()

        X_trainset.loc[segment, 'standard_deviation_rolling_standard_deviation' + str(windows)] = x_roll_std.std()

        X_trainset.loc[segment, 'maximum_rolling_standard_deviation' + str(windows)] = x_roll_std.max()

        X_trainset.loc[segment, 'minimum_rolling_standard_deviation' + str(windows)] = x_roll_std.min()

        X_trainset.loc[segment, 'quantile_1_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.01)

        X_trainset.loc[segment, 'quantile_5_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.05)

        X_trainset.loc[segment, 'quantile_95_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.95)

        X_trainset.loc[segment, 'quantile_99_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.99)

        X_trainset.loc[segment, 'average_change_absolute_rolling_standard_deviation' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_trainset.loc[segment, 'average_change_rate_rolling_standard_deviation' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_trainset.loc[segment, 'maximum_absolute_rolling_standard_deviation' + str(windows)] = np.abs(x_roll_std).max()

        

        X_trainset.loc[segment, 'average_absolute_rolling_mean' + str(windows)] = x_roll_mean.mean()

        X_trainset.loc[segment, 'standard_deviation_rolling_mean' + str(windows)] = x_roll_mean.std()

        X_trainset.loc[segment, 'maximum_rolling_mean' + str(windows)] = x_roll_mean.max()

        X_trainset.loc[segment, 'minimum_rolling_mean' + str(windows)] = x_roll_mean.min()

        X_trainset.loc[segment, 'quantile_1_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.01)

        X_trainset.loc[segment, 'quantile_5_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.05)

        X_trainset.loc[segment, 'quantile_95_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.95)

        X_trainset.loc[segment, 'quantile_99_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.99)

        X_trainset.loc[segment, 'average_change_absolute_rolling_mean' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_trainset.loc[segment, 'average_change_rate_rolling_mean' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_trainset.loc[segment, 'maximum_absolute_rolling_mean' + str(windows)] = np.abs(x_roll_mean).max()

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_testset = pd.DataFrame(columns=X_trainset.columns, dtype=np.float64, index=submission.index)



for i, seg_id in enumerate(tqdm_notebook(X_testset.index)):

    each_seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x_rawdata = each_seg['acoustic_data']

    x_roll = x_rawdata.rolling(windows).std().dropna().values

    x = x_rawdata.values

    

    X_testset.loc[seg_id, 'average'] = x.mean()

    X_testset.loc[seg_id, 'standard_deviation'] = x.std()

    X_testset.loc[seg_id, 'maximum'] = x.max()

    X_testset.loc[seg_id, 'minimum'] = x.min()

    X_testset.loc[seg_id, 'quantile_1_percentile'] = np.quantile(x,0.01)

    X_testset.loc[seg_id, 'quantile_5_percentile'] = np.quantile(x,0.05)

    X_testset.loc[seg_id, 'quantile_95_percentile'] = np.quantile(x,0.95)

    X_testset.loc[seg_id, 'quantile_99_percentile'] = np.quantile(x,0.99)

    X_testset.loc[seg_id, 'median_absolute'] = np.median(np.abs(x))

    X_testset.loc[seg_id, 'quantile_95_percentile_absolute'] = np.quantile(np.abs(x),0.95)

    X_testset.loc[seg_id, 'quantile_99_percentile_absolute'] = np.quantile(np.abs(x),0.99)

    X_testset.loc[seg_id, 'F_test_measure'], X_trainset.loc[segment, 'p_test_measure'] = stats.f_oneway(x[:30000],x[30000:60000],x[60000:90000],x[90000:120000],x[120000:])

    X_testset.loc[seg_id, 'average_change_absolute'] = np.mean(np.diff(x))

    X_testset.loc[seg_id, 'average_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_testset.loc[seg_id, 'maximum_absolute'] = np.abs(x).max()

    

    for windows in [10,100]:

        x_roll_std = x_rawdata.rolling(windows).std().dropna().values

        x_roll_mean = x_rawdata.rolling(windows).mean().dropna().values

        

        X_testset.loc[seg_id, 'average_rolling_standard_deviation' + str(windows)] = x_roll_std.mean()

        X_testset.loc[seg_id, 'standard_deviation_rolling_standard_deviation' + str(windows)] = x_roll_std.std()

        X_testset.loc[seg_id, 'maximum_rolling_standard_deviation' + str(windows)] = x_roll_std.max()

        X_testset.loc[seg_id, 'minimum_rolling_standard_deviation' + str(windows)] = x_roll_std.min()

        X_testset.loc[seg_id, 'quantile_1_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.01)

        X_testset.loc[seg_id, 'quantile_5_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.05)

        X_testset.loc[seg_id, 'quantile_95_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.95)

        X_testset.loc[seg_id, 'quantile_99_percentile_rolling_standard_deviation' + str(windows)] = np.quantile(x_roll_std,0.99)

        X_testset.loc[seg_id, 'average_change_absolute_rolling_standard_deviation' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_testset.loc[seg_id, 'average_change_rate_rolling_standard_deviation' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_testset.loc[seg_id, 'maximum_absolute_rolling_standard_deviation' + str(windows)] = np.abs(x_roll_std).max()

        

        X_testset.loc[seg_id, 'average_absolute_rolling_mean' + str(windows)] = x_roll_mean.mean()

        X_testset.loc[seg_id, 'standard_deviation_rolling_mean' + str(windows)] = x_roll_mean.std()

        X_testset.loc[seg_id, 'maximum_rolling_mean' + str(windows)] = x_roll_mean.max()

        X_testset.loc[seg_id, 'minimum_rolling_mean' + str(windows)] = x_roll_mean.min()

        X_testset.loc[seg_id, 'quantile_1_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.01)

        X_testset.loc[seg_id, 'quantile_5_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.05)

        X_testset.loc[seg_id, 'quantile_95_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.95)

        X_testset.loc[seg_id, 'quantile_99_percentile_rolling_mean' + str(windows)] = np.quantile(x_roll_mean,0.99)

        X_testset.loc[seg_id, 'average_change_absolute_rolling_mean' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_testset.loc[seg_id, 'average_change_rate_rolling_mean' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_testset.loc[seg_id, 'maximum_absolute_rolling_mean' + str(windows)] = np.abs(x_roll_mean).max()
num_folds = 5

k_folds = KFold(n_splits=num_folds, shuffle=True, random_state=11)
def train_model_lgb(X=X_trainset, X_testset=X_testset, y=y_trainset, params=None, k_folds=k_folds, model=None):



  x_values = np.zeros(len(X))

  prediction = np.zeros(len(X_testset))

  scores = []

  feature_importance = pd.DataFrame()

  for fold_n, (trainset_index, valid_set_index) in enumerate(k_folds.split(X)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train_per_fold, X_valid_per_fold = X.iloc[trainset_index], X.iloc[valid_set_index]

    y_train_per_fold, y_valid_per_fold = y.iloc[trainset_index], y.iloc[valid_set_index]



    model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    model.fit(X_train_per_fold, y_train_per_fold, 

           eval_set=[(X_train_per_fold, y_train_per_fold), (X_valid_per_fold, y_valid_per_fold)], eval_metric='mae',

           verbose=1000, early_stopping_rounds=200)



    y_pred_valid = model.predict(X_valid_per_fold)

    y_pred = model.predict(X_testset, num_iteration=model.best_iteration_)



    x_values[valid_set_index] = y_pred_valid.reshape(-1,)

    scores.append(mean_absolute_error(y_valid_per_fold, y_pred_valid))



    prediction += y_pred



  prediction /= num_folds

  print('CV mean score: {0:.4f}.'.format(mean_absolute_error(y, x_values)))

  return x_values, prediction



def train_model_xgb(X=X_trainset, X_testset=X_testset, y=y_trainset, params=None, k_folds=k_folds, model=None):



  x_value = np.zeros(len(X))

  prediction = np.zeros(len(X_testset))

  scores = []

  feature_importance = pd.DataFrame()

  for fold_n, (trainset_index, valid_set_index) in enumerate(k_folds.split(X)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train_per_fold, X_valid_per_fold = X.iloc[trainset_index], X.iloc[valid_set_index]

    y_train_per_fold, y_valid_per_fold = y.iloc[trainset_index], y.iloc[valid_set_index]



    train_data = xgb.DMatrix(data=X_train_per_fold, label=y_train_per_fold, feature_names=X_trainset.columns)

    valid_data = xgb.DMatrix(data=X_valid_per_fold, label=y_valid_per_fold, feature_names=X_trainset.columns)



    watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

    model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

    y_pred_valid = model.predict(xgb.DMatrix(X_valid_per_fold, feature_names=X_trainset.columns), ntree_limit=model.best_ntree_limit)

    y_pred = model.predict(xgb.DMatrix(X_testset, feature_names=X_trainset.columns), ntree_limit=model.best_ntree_limit)



    x_value[valid_set_index] = y_pred_valid.reshape(-1,)

    scores.append(mean_absolute_error(y_valid_per_fold, y_pred_valid))



    prediction += y_pred



  prediction /= num_folds

  print('CV mean score: {0:.4f}.'.format(mean_absolute_error(y, x_value)))

  return x_value, prediction
lgb_params = {'num_leaves': 64,

         'min_data_in_leaf': 50,

         'objective': 'mae',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

          "feature_fraction": 0.5,

         "bagging_freq": 2,

         "bagging_fraction": 0.5,

         "bagging_seed": 0,

         "metric": 'mae',

         "verbosity": -1,

         'reg_alpha': 1.0,

         'reg_lambda': 1.0,

         }

x_value_lgb, prediction_lgb = train_model_lgb(params = lgb_params)
xgb_params = {'eta': 0.01,

              'max_depth': 6,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'colsample_bylevel': 0.8,

              'colsample_bynode': 0.8,

              'lambda': 0.1,

              'alpha' : 0.1,

              'objective': 'reg:linear',

              'eval_metric': 'mae',

              'silent': True,

              'nthread': 4}

x_value_xgb, prediction_xgb = train_model_xgb(params=xgb_params)
print(mean_absolute_error(y_trainset, (x_value_lgb)))
prediction_lgb[:10]
submission['time_to_failure'] = (prediction_lgb)

print(submission.head())

submission.to_csv('submission_all.csv')