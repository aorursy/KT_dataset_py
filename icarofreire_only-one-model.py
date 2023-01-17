# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import gc

warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        if col!='open_channels':

            col_type = df[col].dtypes

            if col_type in numerics:

                c_min = df[col].min()

                c_max = df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

def read_data():

    print('loading and preparing the data')

    train1 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_train1.csv')

    train1 = reduce_mem_usage(train1)

    train2 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_train2.csv')

    train2 = reduce_mem_usage(train2)

    train3 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_train3.csv')

    train3 = reduce_mem_usage(train3)

    

    train = pd.concat([train1, train2, train3], axis = 1)

    del train1, train2, train3

    gc.collect()

    print('train data loaded')

    

    test1 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_test1.csv')

    test1 = reduce_mem_usage(test1)

    test2 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_test2.csv')

    test2 = reduce_mem_usage(test2)

    test3 = pd.read_csv('/kaggle/input/best-filter-and-featureengineering/final_test3.csv')

    test3 = reduce_mem_usage(test3)

    

    test = pd.concat([test1, test2, test3], axis = 1)

    del test1, test2, test3

    gc.collect()

    print('test data loaded')

    

    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

    

    return train, test, submission
train, test, submission = read_data()

print(f'Train data have {train.shape[0]} rows and {train.shape[1]} columns.')

print(f'Test data have {test.shape[0]} rows and {test.shape[1]} columns.')



y_train = train['open_channels']

del train['open_channels']

gc.collect()



print(f'  train.shape =', train.shape)

print('y_train.shape =', y_train.shape)

print(f'   test.shape =', test.shape)
def run_lgb(train, test, y_train, params):

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    oof_pred = np.zeros(len(train))

    y_pred = np.zeros(len(test))

     

    for fold, (tr_ind, val_ind) in enumerate(kf.split(train, y_train)):

        x_train, x_val = train.iloc[tr_ind], train.iloc[val_ind]

        y_trainlgb, y_val = y_train[tr_ind], y_train[val_ind]

        train_set = lgb.Dataset(x_train, y_trainlgb)

        val_set = lgb.Dataset(x_val, y_val)

        

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                         valid_sets = [train_set, val_set], verbose_eval = 100)

        

        oof_pred[val_ind] = model.predict(x_val)

        

        y_pred += model.predict(test) / kf.n_splits

        

    rmse_score = np.sqrt(metrics.mean_squared_error(y_train, oof_pred))

    # want to clip and then round predictions (you can get a better performance using optimization to found the best cuts)

    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)

    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

    f1 = metrics.f1_score(y_train, oof_pred, average = 'macro')

    

    

    print(f'Our oof rmse score is {rmse_score}')

    print(f'Our oof macro f1 score is {f1}')

    return round_y_pred
params = {'boosting_type': 'gbdt',

          'metric': 'rmse',

          'objective': 'regression',

          'n_jobs': -1,

          'seed': 236,

          'num_leaves': 280,

          'learning_rate': 0.026623466966581126,

          'max_depth': 73,

          'lambda_l1': 2.959759088169741,

          'lambda_l2': 1.331172832164913,

          'bagging_fraction': 0.9655406551472153,

          'bagging_freq': 9,

          'colsample_bytree': 0.6867118652742716}



# run model and predict

round_y_pred = run_lgb(train, test, y_train, params)

submission['open_channels'] = round_y_pred

submission.to_csv('submission.csv', index = False)