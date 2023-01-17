import numpy as np 

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc

warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def read_data():

    print('Reading training, testing and submission data...')

    train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

    test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

    print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test, submission



train, test, submission = read_data()



# concatenate data

batch = 50

total_batches = 14

train['set'] = 'train'

test['set'] = 'test'

data = pd.concat([train, test])

for i in range(int(total_batches)):

    data.loc[(data['time'] > i * batch) & (data['time'] <= (i + 1) * batch), 'batch'] = i + 1

train = data[data['set'] == 'train']

test = data[data['set'] == 'test']

del data
# add a lot of features

def preprocess(train, test):

    

    pre_train = train.copy()

    pre_test = test.copy()

    

    for df in [pre_train, pre_test]:

        for window in [10000, 20000, 30000, 40000]:

            # roll backwards

            df['signalmean_t' + str(window)] = df.groupby(['batch'])['signal'].shift(1).rolling(window).mean()

            df['signalvar_t' + str(window)] = df.groupby(['batch'])['signal'].shift(1).rolling(window).var()

            df['signalstd_t' + str(window)] = df.groupby(['batch'])['signal'].shift(1).rolling(window).std()

            df['signalmin_t' + str(window)] = df.groupby(['batch'])['signal'].shift(1).rolling(window).min()

            df['signalmax_t' + str(window)] = df.groupby(['batch'])['signal'].shift(1).rolling(window).max()



            min_max = (df['signal'] - df['signalmin_t' + str(window)]) / (df['signalmax_t' + str(window)] - df['signalmin_t' + str(window)])

            df['norm_t' + str(window)] = min_max * (np.floor(df['signalmax_t' + str(window)]) - np.ceil(df['signalmin_t' + str(window)]))



            # roll forward

            df['signalmean_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].shift(- window - 1).rolling(window).mean()

            df['signalvar_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].shift(- window - 1).rolling(window).var()

            df['signalstd_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].shift(- window - 1).rolling(window).std()

            df['signalmin_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].shift(- window - 1).rolling(window).min()

            df['signalmax_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].shift(- window - 1).rolling(window).max()



            min_max = (df['signal'] - df['signalmin_t' + str(window) + '_lead']) / (df['signalmax_t' + str(window) + '_lead'] - df['signalmin_t' + str(window) + '_lead'])

            df['norm_t' + str(window) + '_lead'] = min_max * (np.floor(df['signalmax_t' + str(window) + '_lead']) - np.ceil(df['signalmin_t' + str(window) + '_lead']))        



    del train, test, min_max

    

    return pre_train, pre_test



# feature engineering

pre_train, pre_test = preprocess(train, test)

del train, test

gc.collect()
def run_lgb(pre_train, pre_test, features, params, get_sample = True, get_metrics = True):

    

    pre_train = pre_train.copy()

    pre_test = pre_test.copy()

    

    # get a random sample for faster training

    if get_sample:

        pre_train = pre_train.sample(frac = 0.1, random_state = 20)

    

    pre_train.reset_index(drop = True, inplace = True)

    pre_test.reset_index(drop = True, inplace = True)

    

    target = 'open_channels'

    

    x_train, x_val, y_train, y_val = train_test_split(pre_train[features], pre_train[target], stratify = pre_train[target], 

                                                      random_state = 42)

    train_set = lgb.Dataset(x_train, y_train)

    val_set = lgb.Dataset(x_val, y_val)

        

    model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                      valid_sets = [train_set, val_set], verbose_eval = 2000)

    

    val_pred = model.predict(x_val) 

    y_pred = model.predict(pre_test[features])

        

    rmse_score = np.sqrt(metrics.mean_squared_error(y_val, val_pred))

    # want to clip and then round predictions

    val_pred = np.round(np.clip(val_pred, 0, 10)).astype(int)

    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

    f1 = metrics.f1_score(y_val, val_pred, average = 'macro')

    if get_metrics:

        print(f'Our val f1_score is {f1}')

        print(f'Our rmse score is {rmse_score}')

    

    return f1, round_y_pred, y_pred
# define hyperparammeter (some random hyperparammeters)

params = {'learning_rate': 0.2, 

          'feature_fraction': 0.75, 

          'bagging_fraction': 0.75,

          'bagging_freq': 1,

          'n_jobs': -1, 

          'seed': 50,

          'metric': 'rmse'

        }



# trian and predict the test set with all the features that were build

features = [col for col in pre_train.columns if col not in ['open_channels', 'time', 'batch', 'set']]

print(f'Training with {len(features)} features')

test_f1, test_r_pred, test_pred = run_lgb(pre_train, pre_test, features, params, get_sample = False, get_metrics = True)
# leave one batch out, trian with 9 (predict all 10 batches as our out of folds)

def val_strat(pre_train, pre_test, features, params, get_sample = True, get_metrics = True):

    

    y_train = pre_train['open_channels']

    

    f1_1, r_y_1, y_1 = run_lgb(pre_train[pre_train['batch']!=1], pre_train[pre_train['batch']==1], features, params, get_sample, get_metrics)

    f1_2, r_y_2, y_2 = run_lgb(pre_train[pre_train['batch']!=2], pre_train[pre_train['batch']==2], features, params, get_sample, get_metrics)

    f1_3, r_y_3, y_3 = run_lgb(pre_train[pre_train['batch']!=3], pre_train[pre_train['batch']==3], features, params, get_sample, get_metrics)

    f1_4, r_y_4, y_4 = run_lgb(pre_train[pre_train['batch']!=4], pre_train[pre_train['batch']==4], features, params, get_sample, get_metrics)

    f1_5, r_y_5, y_5 = run_lgb(pre_train[pre_train['batch']!=5], pre_train[pre_train['batch']==5], features, params, get_sample, get_metrics)

    f1_6, r_y_6, y_6 = run_lgb(pre_train[pre_train['batch']!=6], pre_train[pre_train['batch']==6], features, params, get_sample, get_metrics)

    f1_7, r_y_7, y_7 = run_lgb(pre_train[pre_train['batch']!=7], pre_train[pre_train['batch']==7], features, params, get_sample, get_metrics)

    f1_8, r_y_8, y_8 = run_lgb(pre_train[pre_train['batch']!=8], pre_train[pre_train['batch']==8], features, params, get_sample, get_metrics)

    f1_9, r_y_9, y_9 = run_lgb(pre_train[pre_train['batch']!=9], pre_train[pre_train['batch']==9], features, params, get_sample, get_metrics)

    f1_10, r_y_10, y_10 = run_lgb(pre_train[pre_train['batch']!=10], pre_train[pre_train['batch']==10], features, params, get_sample, get_metrics)

    

    round_y_pred = np.hstack([r_y_1, r_y_2, r_y_3, r_y_4, r_y_5, r_y_6, r_y_7, r_y_8, r_y_9, r_y_10])

    y_pred = np.hstack([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])

    macro_mean_f1_score = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5 + f1_6 + f1_7 + f1_8 + f1_9 + f1_10) / 10

    

    macro_f1_score = metrics.f1_score(y_train, round_y_pred, average = 'macro')

    rmse_score = np.sqrt(metrics.mean_squared_error(y_train, y_pred))

    print(f'Our mean macro f1 score for the 10 folds is {macro_mean_f1_score}')

    print(f'Our out of folds macro f1 score is {macro_f1_score}')

    print(f'Our out of folds rmse score is {rmse_score}')
# using a sample of 10% for demonstration purpose (similar results)

val_strat(pre_train, pre_test, features, params, get_sample = True, get_metrics = False)
submission.open_channels = test_r_pred

submission.to_csv("submission.csv",index=False)