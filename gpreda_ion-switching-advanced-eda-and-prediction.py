import os

import gc

import numpy as np

import pandas as pd

import time

import math

from numba import jit

from math import log, floor

import scipy

from scipy import signal

from scipy.signal import butter, deconvolve

from sklearn.neighbors import KDTree

from pathlib import Path

from sklearn.utils import shuffle

import seaborn as sns

from matplotlib import colors

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize



import lightgbm as lgb

from tqdm import tqdm

from sklearn.model_selection import GroupKFold, KFold

from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score
DATA_PATH = "../input/liverpool-ion-switching"



train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

submission_df = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
print(f"Train data: {train_df.shape}")

print(f"Test data: {test_df.shape}")
train_df.head()
test_df.head()
train_df.describe()
test_df.describe()
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 

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

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df
def get_stats(df):

    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])

    for col in df.columns:

        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]

    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]

    return stats



def print_header():

    print('col         conversion        dtype    na    uniq  size')

    print()

    

def print_values(name, conversion, col):

    template = '{:10}  {:16}  {:>7}  {:2}  {:6}  {:1.2f}MB'

    print(template.format(name, conversion, str(col.dtypes), col.isna().sum(), col.nunique(dropna=False), col.memory_usage(deep=True, index=False) / 1024 ** 2))
get_stats(train_df)
get_stats(test_df)
get_stats(submission_df)
def plot_time_data(data_df, title="Time variation data", color='b'):

    plt.figure(figsize=(18,8))

    plt.plot(data_df["time"], data_df["signal"], color=color)

    plt.title(title, fontsize=24)

    plt.xlabel("Time [sec]", fontsize=20)

    plt.ylabel("Signal", fontsize=20)

    plt.show()
plot_time_data(train_df,"Train data",'g')
plot_time_data(test_df,"Test data",'m')
plot_time_data(train_df[0:500],"Train data",'b')
plot_time_data(train_df[7000:7500],"Train data",'g')
plot_time_data(train_df[8000:8500],"Train data",'b')
plot_time_data(train_df[9000:9500],"Train data",'g')
plot_time_data(train_df[12000:12500],"Train data",'b')
plot_time_data(train_df[15000:15500],"Train data",'r')
plot_time_data(train_df[16000:16500],"Train data",'b')
def plot_time_channel_data(data_df, title="Time variation data"):

    plt.figure(figsize=(18,8))

    plt.plot(data_df["time"], data_df["signal"], color='b', label='Signal')

    plt.plot(data_df["time"], data_df["open_channels"], color='r', label='Open channel')

    plt.title(title, fontsize=24)

    plt.xlabel("Time [sec]", fontsize=20)

    plt.ylabel("Signal & Open channel data", fontsize=20)

    plt.legend(loc='upper right')

    plt.grid(True)

    plt.show()
plot_time_channel_data(train_df[0:500],'Train data: signal & open channel data')
plot_time_channel_data(train_df[7000:7500],'Train data: signal & open channel data (0.7-0.75 sec.)')
plot_time_channel_data(train_df[8000:8500],'Train data: signal & open channel data (0.8-0.85 sec.)')
plot_time_channel_data(train_df[9000:9500],'Train data: signal & open channel data (0.9-0.95 sec.)')
plot_time_channel_data(train_df[15000:15500],'Train data: signal & open channel data (1.5-1.55 sec.)')
plot_time_channel_data(train_df[16000:16500],'Train data: signal & open channel data (1.6-1.65 sec.)')
plot_time_channel_data(train_df[1200000:1200500],'Train data: signal & open channel data (120-120.05 sec.)')
plot_time_channel_data(train_df[1300000:1300500],'Train data: signal & open channel data (130-130.05 sec.)')
plot_time_channel_data(train_df[1400000:1400500],'Train data: signal & open channel data (140-140.05 sec.)')
plot_time_channel_data(train_df[1500000:1500500],'Train data: signal & open channel data (150-150.05 sec.)')
plot_time_channel_data(train_df[3500000:3500500],'Train data: signal & open channel data (350-350.05 sec.)')
plot_time_channel_data(train_df[4100000:4100500],'Train data: signal & open channel data (410-410.05 sec.)')
plot_time_channel_data(train_df[4500000:4500500],'Train data: signal & open channel data (450-450.05 sec.)')
def plot_open_channel_count(data_df, title):

    plt.figure(figsize=(8,6))

    sns.countplot(data_df['open_channels'])

    plt.title(title)

    plt.show()
plot_open_channel_count(train_df,'Open channels distribution')
plot_open_channel_count(train_df[0:1000000],'Open channels distribution (0-100 sec.)')
plot_open_channel_count(train_df[1000000:2000000],'Open channels distribution (100-200 sec.)')
plot_open_channel_count(train_df[2000000:3000000],'Open channels distribution (200-300 sec.)')
plot_open_channel_count(train_df[3000000:4000000],'Open channels distribution (300-400 sec.)')
plot_open_channel_count(train_df[4000000:5000000],'Open channels distribution (400-500 sec.)')
def average_signal_smoothing(signal, kernel_size=3, stride=1):

    sample = []

    start = 0

    end = kernel_size

    while end <= len(signal):

        start = start + stride

        end = end + stride

        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))

    return np.array(sample[::kernel_size])
def plot_signal_signal_smoothed_open_channel(data_df, title):

    sm_df = average_signal_smoothing(data_df["signal"])

    plt.figure(figsize=(18,8))

    plt.plot(data_df["time"], data_df["signal"], color='b', label='Signal')

    plt.plot(data_df["time"][2:], sm_df, color='g', label='Smoothed signal')

    plt.plot(data_df["time"], data_df["open_channels"], color='r', label='Open channel')

    plt.title(title, fontsize=24)

    plt.xlabel("Time [sec]", fontsize=20)

    plt.ylabel("Signal & Open channel data", fontsize=20)

    plt.legend(loc='upper right')

    plt.grid(True)

    plt.show()    
plot_signal_signal_smoothed_open_channel(train_df[0:200], "Train data: signal, smoothed signal & open channel data (0-0.02 sec.)")
plot_signal_signal_smoothed_open_channel(train_df[0:200], "Train data: signal, smoothed signal & open channel data (0-0.02 sec.)")
plot_signal_signal_smoothed_open_channel(train_df[7200:7400], "Train data: signal, smoothed signal & open channel data (0.72-0.74 sec.)")
plot_signal_signal_smoothed_open_channel(train_df[15300:15500], "Train data: signal, smoothed signal & open channel data (1.53-1.55 sec.)")
plot_signal_signal_smoothed_open_channel(train_df[3500000:3500200],'Train data: signal, smoothed signal & open channel data (350-350.02 sec.)')
train_df['train'] = True

test_df['train'] = False

all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

all_data['train'] = all_data['train'].astype('bool')



all_data = all_data.sort_values(by=['time']).reset_index(drop=True)

all_data.index = ((all_data.time * 10_000) - 1).values

all_data['batch'] = all_data.index // 50_000

all_data['batch_index'] = all_data.index  - (all_data.batch * 50_000)

all_data['batch_slices'] = all_data['batch_index']  // 5_000

all_data['batch_slices2'] = all_data.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)



# 50_000 Batch Features

all_data['signal_batch_min'] = all_data.groupby('batch')['signal'].transform('min')

all_data['signal_batch_max'] = all_data.groupby('batch')['signal'].transform('max')

all_data['signal_batch_std'] = all_data.groupby('batch')['signal'].transform('std')

all_data['signal_batch_mean'] = all_data.groupby('batch')['signal'].transform('mean')

all_data['signal_batch_median'] = all_data.groupby('batch')['signal'].transform('median')

#all_data['signal_batch_mad'] = all_data.groupby('batch')['signal'].transform('mad')

all_data['signal_batch_skew'] = all_data.groupby('batch')['signal'].transform('skew')

all_data['mean_abs_chg_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))

all_data['median_abs_chg_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.median(np.abs(np.diff(x))))

all_data['abs_max_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))

all_data['abs_min_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))

all_data['abs_mean_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.mean(np.abs(x)))

all_data['abs_median_batch'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.median(np.abs(x)))

all_data['moving_average_batch_1000_mean'] = all_data.groupby(['batch'])['signal'].rolling(window=1000).mean().mean(skipna=True)





all_data['range_batch'] = all_data['signal_batch_max'] - all_data['signal_batch_min']

all_data['maxtomin_batch'] = all_data['signal_batch_max'] / all_data['signal_batch_min']

all_data['abs_avg_batch'] = (all_data['abs_min_batch'] + all_data['abs_max_batch']) / 2



# 5_000 Batch Features

all_data['signal_batch_5k_min'] = all_data.groupby('batch_slices2')['signal'].transform('min')

all_data['signal_batch_5k_max'] = all_data.groupby('batch_slices2')['signal'].transform('max')

all_data['signal_batch_5k_std'] = all_data.groupby('batch_slices2')['signal'].transform('std')

all_data['signal_batch_5k_mean'] = all_data.groupby('batch_slices2')['signal'].transform('mean')

all_data['signal_batch_5k_median'] = all_data.groupby('batch_slices2')['signal'].transform('median')

all_data['signal_batch_5k_mad'] = all_data.groupby('batch_slices2')['signal'].transform('mad')

all_data['mean_abs_chg_batch_5k'] = all_data.groupby(['batch_slices2'])['signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))

all_data['median_abs_chg_batch_5k'] = all_data.groupby(['batch'])['signal'].transform(lambda x: np.median(np.abs(np.diff(x))))

all_data['abs_max_batch_5k'] = all_data.groupby(['batch_slices2'])['signal'].transform(lambda x: np.max(np.abs(x)))

all_data['abs_min_batch_5k'] = all_data.groupby(['batch_slices2'])['signal'].transform(lambda x: np.min(np.abs(x)))

all_data['abs_mean_batch_5k'] = all_data.groupby(['batch_slices2'])['signal'].transform(lambda x: np.mean(np.abs(x)))

all_data['abs_median_batch_5k'] = all_data.groupby(['batch_slices2'])['signal'].transform(lambda x: np.median(np.abs(x)))



all_data['moving_average_batch_5k_1000_mean'] = all_data.groupby(['batch_slices2'])['signal'].rolling(window=1000).mean().mean(skipna=True)



all_data['range_batch_5k'] = all_data['signal_batch_5k_max'] - all_data['signal_batch_5k_min']

all_data['maxtomin_batch_5k'] = all_data['signal_batch_5k_max'] / all_data['signal_batch_5k_min']

all_data['abs_avg_batch_5k'] = (all_data['abs_min_batch_5k'] + all_data['abs_max_batch_5k']) / 2



#add shifts

all_data['signal_shift+1'] = all_data.groupby(['batch']).shift(1)['signal']

all_data['signal_shift-1'] = all_data.groupby(['batch']).shift(-1)['signal']

all_data['signal_shift+2'] = all_data.groupby(['batch']).shift(2)['signal']

all_data['signal_shift-2'] = all_data.groupby(['batch']).shift(-2)['signal']

all_data['signal_shift+1_5k'] = all_data.groupby(['batch_slices2']).shift(1)['signal']

all_data['signal_shift-1_5k'] = all_data.groupby(['batch_slices2']).shift(-1)['signal']

all_data['signal_shift+2_5k'] = all_data.groupby(['batch_slices2']).shift(2)['signal']

all_data['signal_shift-2_5k'] = all_data.groupby(['batch_slices2']).shift(-2)['signal']



all_data['abs_max_signal_shift+1_5k'] = all_data['signal_shift+1_5k'].transform(lambda x: np.max(np.abs(x)))

all_data['abs_max_signal_shift-1_5k'] = all_data['signal_shift-1_5k'].transform(lambda x: np.max(np.abs(x)))

for c in ['signal_batch_mean','signal_batch_median',

          'mean_abs_chg_batch','abs_max_batch','abs_min_batch', 'abs_median_batch',

          'moving_average_batch_1000_mean', 

          'range_batch','abs_avg_batch',

          'signal_batch_5k_mean', 'signal_batch_5k_median', 'signal_batch_5k_mad', 

          'moving_average_batch_5k_1000_mean', 

          'mean_abs_chg_batch_5k','abs_max_batch_5k', 'abs_min_batch_5k', 'abs_median_batch_5k', 

          'range_batch_5k','abs_avg_batch_5k',

          'signal_shift+1', 'signal_shift+2', 'signal_shift-1', 'signal_shift-2', #'signal_shift+3','signal_shift-3',

          'signal_shift+1_5k', 'signal_shift+2_5k', 'signal_shift-1_5k', 'signal_shift-2_5k', #'signal_shift+3_5k','signal_shift-3_5k',

          'abs_max_signal_shift+1_5k', 'abs_max_signal_shift-1_5k'

         ]:

    all_data[f'{c}_msignal'] = all_data[c] - all_data['signal']
FEATURES = [f for f in all_data.columns if f not in ['open_channels','index','time','train','batch',

                                                     'signal_batch_max', 'signal_batch_mad', 'maxtomin_batch'

                                                    'batch_index','batch_slices','batch_slices2', 'median_abs_chg_batch_5k',

                                                    'abs_mean_batch', 'abs_median_batch', 'abs_avg_batch', 'signal_batch_median',

                                                     'abs_max_batch', 'abs_max_batch_5k', 'abs_median_batch_5k', 

                                                    'moving_average_batch_1000_mean', 'moving_average_batch_5k_1000_mean']]

print('....: FEATURE LIST :....')

print([f for f in FEATURES])

print(f"Features: {len(FEATURES)}")
get_stats(all_data)
TARGET = 'open_channels'

all_data['train'] = all_data['train'].astype('bool')

train_df = all_data.query('train').copy()

test_df = all_data.query('not train').copy()

train_df['open_channels'] = train_df['open_channels'].astype(int)

X = train_df[FEATURES]

X_test = test_df[FEATURES]

y = train_df[TARGET].values

sub = test_df[['time']].copy()

groups = train_df['batch']

oof_df = train_df[['signal','open_channels']].copy()
del all_data

del train_df

del test_df

gc.collect()


TOTAL_FOLDS = 7



MODEL_TYPE = 'LGBM'

SHUFFLE = True

NUM_BOOST_ROUND = 3_500

EARLY_STOPPING_ROUNDS = 50

VERBOSE_EVAL = 500

RANDOM_SEED = 99943





params = {'learning_rate': 0.05,

          'max_depth': -1,

          'num_leaves': 2**8+1,

          'feature_fraction': 0.8,

          'bagging_fraction': 0.8,

          'bagging_freq': 0,

          'n_jobs': 8,

          'seed': RANDOM_SEED,

          'metric': 'rmse',

          'objective' : 'regression',

          'num_class': 1

        }



kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=SHUFFLE, random_state=RANDOM_SEED)



feature_importance_df = pd.DataFrame()



fold_ = 1 

for tr_idx, val_idx in kfold.split(X, y, groups=groups):

    print(f'====== Fold {fold_:0.0f} of {TOTAL_FOLDS} ======')

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]

    y_tr, y_val = y[tr_idx], y[val_idx]

    train_set = lgb.Dataset(X_tr, y_tr)

    val_set = lgb.Dataset(X_val, y_val)



    model = lgb.train(params,

                      train_set,

                      num_boost_round = NUM_BOOST_ROUND,

                      early_stopping_rounds = EARLY_STOPPING_ROUNDS,

                      valid_sets = [train_set, val_set],

                      verbose_eval = VERBOSE_EVAL)



    preds = model.predict(X_val, num_iteration=model.best_iteration)

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    test_preds = model.predict(X_test, num_iteration=model.best_iteration)

    test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)



    oof_df.loc[oof_df.iloc[val_idx].index, 'oof'] = preds

    sub[f'open_channels_fold{fold_}'] = test_preds



    f1 = f1_score(oof_df.loc[oof_df.iloc[val_idx].index]['open_channels'],

                  oof_df.loc[oof_df.iloc[val_idx].index]['oof'],

                            average = 'macro')

    rmse = np.sqrt(mean_squared_error(oof_df.loc[oof_df.index.isin(val_idx)]['open_channels'],

                                      oof_df.loc[oof_df.index.isin(val_idx)]['oof']))



    

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = FEATURES

    fold_importance_df["importance"] = model.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    print(f'Fold {fold_} - validation f1: {f1:0.5f}')

    print(f'Fold {fold_} - validation rmse: {rmse:0.5f}')



    fold_ += 1



oof_f1 = f1_score(oof_df['open_channels'],

                    oof_df['oof'],

                    average = 'macro')

oof_rmse = np.sqrt(mean_squared_error(oof_df['open_channels'],

                                      oof_df['oof']))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:100].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,14))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
s_cols = [s for s in sub.columns if 'open_channels' in s]



sub['open_channels'] = sub[s_cols].median(axis=1).astype(int)

sub[['time','open_channels']].to_csv('./submission.csv',

        index=False,

        float_format='%0.4f')