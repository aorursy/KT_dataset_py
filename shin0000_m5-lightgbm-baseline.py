import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

import os

import gc
from sklearn.preprocessing import LabelEncoder
base_dir = '/kaggle/input/m5-forecasting-accuracy/'

train_dir = os.path.join(base_dir, 'sales_train_evaluation.csv')

test_dir = os.path.join(base_dir, 'sample_submission.csv')

calendar_dir = os.path.join(base_dir, 'calendar.csv')

price_dir = os.path.join(base_dir, 'sell_prices.csv')

sub_dir = os.path.join(base_dir, 'sample_submission.csv')
df_train = pd.read_csv(train_dir)

df_test = pd.read_csv(test_dir)

df_calendar = pd.read_csv(calendar_dir)

df_price = pd.read_csv(price_dir)

df_sub = pd.read_csv(sub_dir)
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
def making_train_data(df_train):

    print("processing train data")

    df_train_after = pd.melt(df_train, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='days', value_name='demand')

    df_train_after['days'] = df_train_after['days'].map(lambda x: int(x[2:]))

    df_train_after = df_train_after.drop(['id'], axis=1)

    df_train_after = reduce_mem_usage(df_train_after)

    gc.collect()

    return df_train_after
def making_test_data(df_test):

    print("processing test data")

    df_test['item_id'] = df_test['id'].map(lambda x: x[:-16])

    df_test['dept_id'] = df_test['item_id'].map(lambda x: x[:-4])

    df_test['cat_id'] = df_test['dept_id'].map(lambda x: x[:-2])

    df_test['store_id'] = df_test['id'].map(lambda x: x[-15:-11])

    df_test['state_id'] = df_test['store_id'].map(lambda x: x[:-2])

    df_test['va_or_ev'] = df_test['id'].map(lambda x: x[-10:])

    df_test_val = df_test.loc[df_test['va_or_ev'] == 'validation', :]

    df_test_ev = df_test.loc[df_test['va_or_ev'] == 'evaluation', :]

    df_test_val_after = pd.melt(df_test_val, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')

    df_test_ev_after = pd.melt(df_test_ev, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')

    df_test_after = pd.concat([df_test_val_after, df_test_ev_after])

    df_test_after['days'] = df_test_after['days'].map(lambda x: int(x[1:]))

    df_test_after.loc[df_test_after['va_or_ev']=='evaluation', ['days']] += 28

    df_test_after['days'] += 1913

    df_test_after = df_test_after.drop(['va_or_ev'], axis=1)

    df_test_after = df_test_after.drop(['id'], axis=1)

    df_test_after = reduce_mem_usage(df_test_after)

    return df_test_after
def making_train_test_data(df_train ,df_test):

    df_train = making_train_data(df_train)

    df_test = making_test_data(df_test)

    print("processing train test data")

    max_train_days = df_train['days'].max()

    min_test_days = df_test['days'].min()

    shift_data = 6

    df_test = pd.concat([df_train.loc[max_train_days - 28 * shift_data <= df_train['days'], :], df_test.loc[df_test['days'] > max_train_days, :]]).reset_index(drop=True)

    

#     shift_days_set = [28, 29, 30]

#     for i in shift_days_set:

#         df_train['demand_{}day_ago'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(i))

#         df_test['demand_{}day_ago'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(i))

#         gc.collect()

        

    rolling_days_set = [2, 3, 5, 7, 14, 28, 56, 140]

    for i in rolling_days_set:

        df_train['demand_{}day_mean'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).mean())

#         df_train['demand_{}day_max'.format(i)] = df_train.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).max())

        

        df_test['demand_{}day_mean'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).mean())

#         df_test['demand_{}day_max'.format(i)] = df_test.groupby(['item_id', 'store_id'])['demand'].transform(lambda x: x.shift(28).rolling(i).max())

        df_train = reduce_mem_usage(df_train)

        df_test = reduce_mem_usage(df_test)

        gc.collect()

    

    df_test = df_test.loc[df_test['days'] >= min_test_days, :]

    df_test = reduce_mem_usage(df_test)

    gc.collect()

    

    return df_train, df_test
def making_calendar_data(df_calendar):

    df_calendar = reduce_mem_usage(df_calendar)

    gc.collect()

    print("processing calendar data")

    df_calendar['days'] = df_calendar['d'].map(lambda x: int(x[2:]))

    event_type = {np.nan: 1, 'Sporting': 2, 'Cultural': 3, 'National': 5, 'Religious': 7}

    df_calendar['event_type_1'] = df_calendar['event_type_1'].map(event_type)

    df_calendar['event_type_2'] = df_calendar['event_type_2'].map(event_type)

    df_calendar['event_type'] = df_calendar['event_type_1'] * df_calendar['event_type_2']

    le = LabelEncoder()

    le.fit(df_calendar['event_type'])

    df_calendar['event_type'] = le.transform(df_calendar['event_type'])

    df_calendar = df_calendar.drop(['event_type_1', 'event_type_2', 'event_name_1', 'event_name_2', 'd', 'weekday', 'date', 'year'], axis=1)

#     df_calendar['event_type_1day_ago'] = df_calendar['event_type'].shift(1)

#     df_calendar['event_type_1day_after'] = df_calendar['event_type'].shift(-1)

    df_calendar = reduce_mem_usage(df_calendar)

    gc.collect()

    return df_calendar
def making_price_data(df_price):

    df_price = reduce_mem_usage(df_price)

    gc.collect()

    print("processing price data")

#     shift_days_set = [28, 35, 42]

#     for i in shift_days_set:

#         df_price['price_{}day_ago'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(i))

#     gc.collect()

    

    rolling_days_set = [28, 140]

    for i in rolling_days_set:

        df_price['price_{}day_mean'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(28).rolling(i).mean())

        df_price['price_{}day_max'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(28).rolling(i).max())

        df_price['price_{}day_min'.format(i)] = df_price.groupby(['item_id', 'store_id'])['sell_price'].transform(lambda x: x.shift(28).rolling(i).min())

        df_price = reduce_mem_usage(df_price)

        gc.collect()

    return df_price
def concat_data(df_train, df_test, df_calendar, df_price):

    df_train, df_test = making_train_test_data(df_train ,df_test)

    df_calendar = making_calendar_data(df_calendar)

    df_price = making_price_data(df_price)

    print("concat data")

    df_train = pd.merge(df_train, df_calendar, on='days', how='left')

    df_test = pd.merge(df_test, df_calendar, on='days', how='left')

    df_train = pd.merge(df_train, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')

    df_test = pd.merge(df_test, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')

    df_train = df_train.drop(['wm_yr_wk'], axis=1)

    df_test = df_test.drop(['wm_yr_wk'], axis=1)

    del df_calendar, df_price

    gc.collect()

    df_train = reduce_mem_usage(df_train)

    df_test = reduce_mem_usage(df_test)

    gc.collect()

    return df_train, df_test
def labeling_data(df_train, df_test, df_calendar, df_price):

    df_train, df_test = concat_data(df_train, df_test, df_calendar, df_price)

    print("labeling data")

    label_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    for c in label_columns:

        le  = LabelEncoder()

        le.fit(df_train[c])

        df_train[c] = le.transform(df_train[c])

        df_test[c] = le.transform(df_test[c])

        if c != 'item_id':

            print(le.classes_)

    

    df_train = reduce_mem_usage(df_train)

    df_test = reduce_mem_usage(df_test)

    gc.collect()

    

    return df_train, df_test
df_train, df_test = labeling_data(df_train, df_test, df_calendar, df_price)
for c in df_train.columns:

    print(c)
df_train
df_test
from sklearn.metrics import mean_squared_error

import lightgbm as lgbm



def metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))



def get_feature_importances(data, store_id, y_valid, y_valid_pred, used_features):

    train_features = used_features

    imp_df = pd.DataFrame()

    imp_df["importance_gain_{}".format(store_id)] = lgb.feature_importance(importance_type='gain')

    imp_df["importance_split_{}".format(store_id)] = lgb.feature_importance(importance_type='split')

    imp_df["valid_rmse_{}".format(store_id)] = mean_squared_error(y_valid, y_valid_pred, squared=False)

    imp_df["valid_wrmse_{}".format(store_id)] = metric(y_valid, y_valid_pred)

    return imp_df



total_imp_df = pd.DataFrame()

df_sub_ensemble = df_test.loc[:, ['item_id', 'store_id', 'days', 'demand']]

df_sub_ensemble['demand_model'] = 0
df_train['snap'] = 0

df_test['snap'] = 0

used_features = [c for c in df_train.columns if c not in ['demand', 'item_id', 'store_id', 'state_id', 'days', 'snap_CA', 'snap_TX', 'snap_WI']]

total_imp_df["feature"] = used_features

store_id_list = df_train['store_id'].unique()

for store_id in store_id_list:

    print('store_id {}/10'.format(store_id + 1))

    

    df_train['snap'] = 0

    df_test['snap'] = 0

    if 0 <= store_id <= 3:

        df_train['snap'] = df_train['snap_CA']

        df_test['snap'] = df_test['snap_CA']

    elif 4 <= store_id <= 6:

        df_train['snap'] = df_train['snap_TX']

        df_test['snap'] = df_test['snap_TX']

    else:

        df_train['snap'] = df_train['snap_WI']

        df_test['snap'] = df_test['snap_WI']

        

    train_index = (df_train['days'] < 1913 - 28) & (df_train['store_id'] == store_id)

    valid_index = (1913 - 28 <= df_train['days']) & (df_train['store_id'] == store_id)

    test_index = (df_test['store_id'] == store_id)

    

    X_train = df_train.loc[train_index, used_features].values

    y_train = df_train.loc[train_index, 'demand'].values



    X_valid = df_train.loc[valid_index, used_features].values

    y_valid = df_train.loc[valid_index, 'demand'].values



    X_test = df_test.loc[test_index, used_features].values



    lgb_params = {

        'objective': 'poisson',

        'num_iterations' : 2000,

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'n_jobs': -1,

        'seed': 42,

        'learning_rate': 0.075,

        'bagging_fraction': 0.75,

        'bagging_freq': 10,

        'colsample_bytree': 0.75

                  }



    train_data = lgbm.Dataset(X_train, y_train)

    valid_data = lgbm.Dataset(X_valid, y_valid)



    lgb = lgbm.train(lgb_params, train_data, valid_sets=[train_data, valid_data], early_stopping_rounds=10, verbose_eval=20)

    

    y_valid_pred = lgb.predict(X_valid, num_iteration=lgb.best_iteration)

    y_test_pred = lgb.predict(X_test, num_iteration=lgb.best_iteration)

    df_sub_ensemble.loc[test_index, ['demand_model']] = y_test_pred

    

    print(metric(y_valid, y_valid_pred))

    

    imp_df = get_feature_importances(df_train, store_id, y_valid, y_valid_pred, used_features)

    total_imp_df = pd.concat([total_imp_df, imp_df], axis=1, sort=False)

    

    del X_train, X_valid, X_test, y_train, y_valid, lgb

    gc.collect()
df_imp = pd.DataFrame(columns=['features', 'importance_gain', 'importance_split', 'valid_rmse', 'valid_wrmse'])

df_imp["features"] = used_features

df_imp['importance_gain'] = 0

df_imp['importance_split'] = 0

df_imp['valid_rmse'] = 0

df_imp['valid_wrmse'] = 0

n_stores = len(store_id_list)

for store_id in store_id_list:

    df_imp['importance_gain'] += total_imp_df['importance_gain_{}'.format(store_id)].values / n_stores

    df_imp['importance_split'] += total_imp_df['importance_split_{}'.format(store_id)].values / n_stores

    df_imp['valid_rmse'] += total_imp_df['valid_rmse_{}'.format(store_id)].values / n_stores

    df_imp['valid_wrmse'] += total_imp_df['valid_wrmse_{}'.format(store_id)].values / n_stores
df_imp.sort_values(by='importance_gain', ascending=False)
df_sub_before = df_test.loc[:, ['days', 'demand']]
df_sub_before['demand'] = df_sub_ensemble['demand_model']
df_sub = pd.read_csv(sub_dir)

df_sub_base = pd.read_csv(sub_dir)
def making_submission(df_sub, df_sub_before, df_sub_base):

    df_sub['va_or_ev'] = df_sub['id'].map(lambda x: x[-10:])

    df_sub_val = df_sub.loc[df_sub['va_or_ev'] == 'validation', :]

    df_sub_ev = df_sub.loc[df_sub['va_or_ev'] == 'evaluation', :]

    df_sub_val = df_sub_val.melt(id_vars=['id', 'va_or_ev'], var_name='days', value_name='demand').drop(['va_or_ev'], axis=1)

    df_sub_ev = df_sub_ev.melt(id_vars=['id', 'va_or_ev'], var_name='days', value_name='demand').drop(['va_or_ev'], axis=1)

    num_va = df_sub_val.shape[0]

    num_ev = df_sub_ev.shape[0]

    df_sub_val['demand'] = df_sub_before['demand'][:num_va].values

    df_sub_ev['demand'] = df_sub_before['demand'][num_va:].values

    df_sub_val = df_sub_val.pivot(index='id', columns='days', values='demand').reset_index()

    df_sub_ev = df_sub_ev.pivot(index='id', columns='days', values='demand').reset_index()

    df_sub_after = pd.concat([df_sub_val, df_sub_ev])

    df_sub_columns = ['id'] + ['F{}'.format(i+1) for i in range(28)]

    df_sub = df_sub_after.loc[:, df_sub_columns]

    df_sub.columns = df_sub_columns

    df_sub = pd.merge(df_sub_base['id'], df_sub, on='id', how='left')

    return df_sub
df_sub = making_submission(df_sub, df_sub_before, df_sub_base)

df_sub.to_csv('./my_submission.csv', index=False)
df_sub