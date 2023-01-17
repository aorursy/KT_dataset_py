import warnings

import time



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor



warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(314)

USING_GPU = True
submission = pd.read_csv('../input/bigquery-geotab-intersection-congestion/sample_submission.csv')

labeled = pd.read_csv('../input/geotabprocessed/train_processed.csv').set_index('RowId')

test = pd.read_csv('../input/geotabprocessed/test_processed.csv').set_index('RowId').loc[:1920334, :]
target_vars = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 

               'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 

               'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']

num_vars = ['dist_to_5pm', 'dist_to_8am', 'latitude_dist', 'longitude_dist']

bool_vars = ['Weekend', 'is_Atlanta', 'is_Boston', 'is_Chicago', 'is_Philadelphia']

cat_vars = [var for var in labeled.columns if var not in target_vars+num_vars+bool_vars]



labeled[cat_vars] = labeled[cat_vars].astype('category')

test[cat_vars] = test[cat_vars].astype('category')

cat_idxs = [i for i, var in enumerate(labeled.drop(columns=target_vars).columns) if var in cat_vars]
mask = np.random.rand(len(labeled)) < 0.9

train = labeled[mask]

val = labeled[~mask]
def get_X_y(df):

    X = df.drop(columns=target_vars)

    y = df[target_vars]

    return X, y
X_train, y_train = get_X_y(train)

X_val, y_val = get_X_y(val)
params = {'cat_features': cat_idxs, 

          'eval_metric': 'RMSE',

          'random_seed':314, 

          'one_hot_max_size':24, 

          'boosting_type':'Plain', 

          'bootstrap_type':'Bayesian',

          'max_ctr_complexity':2, 

          'iterations':10**5, 

          'learning_rate': 0.1,

         }

if USING_GPU:

    params['task_type'] = 'GPU'

    params['border_count'] = 254
def search(params, param_name, param_list, tune_var):

    '''

    Returns a dictionary of tested hyperparameter values and their corresponding scores.

    '''

    scores={}

    for val in param_list:

        params[param_name] = val

        catboost = CatBoostRegressor(**params).fit(

            X_train, y_train[tune_var], early_stopping_rounds=20, 

            eval_set=(X_val,y_val[tune_var]), plot=True)

        pred = catboost.predict(X_val)

        pred[pred < 0] = 0

        scores[val] = mean_squared_error(pred, y_val[tune_var])**0.5

    del params[param_name]

    del param_list

    sns.lineplot(x=list(scores.keys()), y=list(scores.values()), marker='o').set(xlabel='Hyperparameter Value', ylabel='RMSE');

    return scores.copy()
params['depth'] = 10

params['l2_leaf_reg'] = 18
def get_preds(var, params):

    catboost = CatBoostRegressor(**params).fit(

        X_train, y_train[var], eval_set=(X_val, y_val[var]), 

        early_stopping_rounds=50, plot=True, verbose=False)

    return catboost.predict(X_test)
X_labeled, y_labeled = get_X_y(labeled)

X_test, y_test = get_X_y(test)

preds = {}

for idx, var in enumerate(tqdm(target_vars)):

    preds[idx] = get_preds(var, params)
submission['Target'] = pd.DataFrame(preds).stack().to_frame().iloc[:,0].values

submission.loc[submission['Target'] < 0, 'Target'] = 0

submission.to_csv('my_submission.csv', index=False)
target_vars = ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 

               'TotalTimeStopped_p80', 'DistanceToFirstStop_p20', 

               'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(20,40))

bins = list(range(0, 200, 10))

for i, var in enumerate(target_vars):

    sns.distplot(y_labeled[var], bins=bins, kde=False, ax=ax[i, 0]).set_title(var + ' Training Set Labels')

    sns.distplot(preds[i], bins=bins, kde=False, ax=ax[i, 1]).set_title(var + ' Test Set Predictions')