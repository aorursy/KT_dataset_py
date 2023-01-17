import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

from typing import Union

from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing

import gc

import lightgbm as lgb

from datetime import datetime, timedelta

from sklearn.model_selection import GroupKFold

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# helper functions to reduce memory

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

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



# function to read our data

def read_data():

    # read data

    data = pd.read_pickle('/kaggle/input/m5-reduce-data/data_small.pkl')

    # fillna and label encode categorical features

    data = transform(data)

    # read submission

    submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

    return data, submission



# filla na and label encode categorical features

def transform(data):

    

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in nan_features:

        data[feature].fillna('unknown', inplace = True)

        

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 

           'event_name_2', 'event_type_2']

    for feature in cat:

        encoder = preprocessing.LabelEncoder()

        data[feature] = encoder.fit_transform(data[feature])

        

    # reduce memory usage

    data = reduce_mem_usage(data)

    

    return data



# simple feature ingineer function

def simple_fe(data):

    

    data_fe = data[['id', 'demand']]

    

    window = 28

    periods = [7, 15, 30, 90]

    group = data_fe.groupby('id')['demand']

    

    # most recent lag data

    for period in periods:

        data_fe['demand_rolling_mean_t' + str(period)] = group.transform(lambda x: x.shift(window).rolling(period).mean())

        data_fe['demand_rolling_std_t' + str(period)] = group.transform(lambda x: x.shift(window).rolling(period).std())

        

    # reduce memory

    data_fe = reduce_mem_usage(data_fe)

    

    # get time features

    data['date'] = pd.to_datetime(data['date'])

    time_features = ['year', 'month', 'quarter', 'week', 'day', 'dayofweek', 'dayofyear']

    dtype = np.int16

    for time_feature in time_features:

        data[time_feature] = getattr(data['date'].dt, time_feature).astype(dtype)

        

    # concat lag and rolling features with main table

    lag_rolling_features = [col for col in data_fe.columns if col not in ['id', 'demand']]

    data = pd.concat([data, data_fe[lag_rolling_features]], axis = 1)

    

    del data_fe

    gc.collect()



    return data



# define custom loss function

def custom_asymmetric_train(y_pred, y_true):

    y_true = y_true.get_label()

    residual = (y_true - y_pred).astype("float")

    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)

    hess = np.where(residual < 0, 2, 2 * 1.15)

    return grad, hess



# define custom evaluation metric

def custom_asymmetric_valid(y_pred, y_true):

    y_true = y_true.get_label()

    residual = (y_true - y_pred).astype("float")

    loss = np.where(residual < 0, (residual ** 2) , (residual ** 2) * 1.15) 

    return "custom_asymmetric_eval", np.mean(loss), False



# define lgbm simple model

def run_lgb(data, features, cat_features):

    

    # reset_index

    data.reset_index(inplace = True, drop = True)

    

    # going to evaluate with the last 28 days

    x_train = data[data['date'] <= '2016-04-24']

    y_train = x_train['demand']

    test = data[data['date'] >= '2016-04-25']



    # define random hyperparammeters

    params = {

        'boosting_type': 'gbdt',

        'n_jobs': -1,

        'seed': 42,

        'learning_rate': 0.1,

        'bagging_fraction': 0.85,

        'bagging_freq': 1, 

        'colsample_bytree': 0.85,

        'colsample_bynode': 0.85,

        'min_data_per_leaf': 25,

        'num_leaves': 200,

        'lambda_l1': 0.5,

        'lambda_l2': 0.5}

    

    oof = np.zeros(len(x_train))

    preds = np.zeros(len(test))

    

    # GroupKFold by week, month to avoid leakage and overfitting (not entirely sure xD)

    kf = GroupKFold(5)

    # get subgroups for each week, year pair

    group = x_train['week'].astype(str) + '_' + x_train['year'].astype(str)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train, group)):

        print(f'Training fold {fold + 1}')

        train_set = lgb.Dataset(x_train.iloc[trn_idx][features], y_train.iloc[trn_idx], 

                                categorical_feature = cat_features)

        val_set = lgb.Dataset(x_train.iloc[val_idx][features], y_train.iloc[val_idx], 

                              categorical_feature = cat_features)

        

        # train with our custom loss function and evaluation metric

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                          valid_sets = [train_set, val_set], verbose_eval = 50, fobj = custom_asymmetric_train, 

                          feval = custom_asymmetric_valid)

    

        # predict oof

        oof[val_idx] = model.predict(x_train.iloc[val_idx][features])



        # predict test

        preds += model.predict(test[features]) / 5

        

        print('-'*50)

        print('\n')

        

    oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))

    print(f'Our out of folds rmse is {oof_rmse}')

        

    test = test[['id', 'date', 'demand']]

    test['demand'] = preds

    return test



# function to get the predictions in the correct format

def predict(test, submission):

    predictions = pd.pivot(test, index = 'id', columns = 'date', values = 'demand').reset_index()

    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 

    evaluation = submission[submission['id'].isin(evaluation_rows)]



    validation = submission[['id']].merge(predictions, on = 'id')

    final = pd.concat([validation, evaluation])

    final.to_csv('submission_custom_loss.csv', index = False)

    

# this is the main function that will run our entire program

def train_and_evaluate():

    

    # read data

    print('Reading our data...')

    data, submission = read_data()

    

    data['date'] = pd.to_datetime(data['date'])

    # get amount of unique days in our data

    days = abs((data['date'].min() - data['date'].max()).days)

    # how many training data do we need to train with at least 2 years and consider lags

    need = 365 + 365 + 90 + 28

    print(f'We have {(days - 28)} days of training history')

    print(f'we have {(days - 28 - need)} days left')

    if (days - 28 - need) > 0:

        print('We have enought training data, lets continue')

    else:

        print('Get more training data, training can fail')

    

    # simple feature engineer

    print('Running simple feature engineering...')

    data = simple_fe(data)

    print('Removing first 118 days')

    # eliminate the first 118 days of our train data because of lags

    min_date = data['date'].min() + timedelta(days = 118)

    data = data[data['date'] > min_date]

    

    # define our numeric features and categorical features

    features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'year', 

                'month', 'quarter', 'week', 'day', 'dayofweek', 'dayofyear', 'demand_rolling_mean_t7', 'demand_rolling_mean_t15', 'demand_rolling_mean_t30', 'demand_rolling_mean_t90',

                'demand_rolling_std_t7', 'demand_rolling_std_t15', 'demand_rolling_std_t30', 'demand_rolling_std_t90']

    

    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 

                    'event_name_2', 'event_type_2']

    

    print('-'*50)

    print('\n')

    print(f'Training model with {len(features)} features...')

    # run lgbm model with 5 GroupKFold (subgroups by year, month)

    test = run_lgb(data, features, cat_features)

    print('Save predictions...')

    # predict

    predict(test, submission)

        

# run our program

train_and_evaluate()