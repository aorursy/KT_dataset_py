import gc
import os
import random

import numpy as np
import pandas as pd
def random_seed_initialize(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
random_seed_initialize()
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
train_data = pd.read_csv('../input/data-without-drift/train_clean.csv')
test_data  = pd.read_csv('../input/data-without-drift/test_clean.csv')
def set_index(df):
    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    return df
def set_batch_index(df, batch_size1=50_000, batch_size2=5_000):
    df['batch'] = df.index // batch_size1
    df['batch_index'] = df.index - (df.batch * batch_size1)
    df['batch_slices'] = df['batch_index'] // batch_size2
    df['batch_slices2'] = df.apply(lambda r: '_'.join(
        [str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    return df
def set_features_batch50000(df):
    df['signal_batch_min'] = df.groupby('batch')['signal'].transform('min')  # 最小値
    df['signal_batch_max'] = df.groupby('batch')['signal'].transform('max')  # 最大値
    df['signal_batch_std'] = df.groupby('batch')['signal'].transform('std')  # 標準偏差
    df['signal_batch_mean'] = df.groupby('batch')['signal'].transform('mean')  # 平均
    df['mean_abs_chg_batch'] = df.groupby(['batch'])['signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))  # 前回との差分の平均
    df['abs_max_batch'] = df.groupby(['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))  # 絶対値の最大値
    df['abs_min_batch'] =df.groupby(['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))  # 絶対値の最小値

    df['range_batch'] = df['signal_batch_max'] - df['signal_batch_min']  # 最大値と最小値のギャップ
    df['maxtomin_batch'] = df['signal_batch_max'] / df['signal_batch_min']  # 最大値÷最小値
    df['abs_avg_batch'] = (df['abs_min_batch'] + df['abs_max_batch']) / 2  # 最大値（絶対値）と最小値（絶対値）の平均
    return df
def set_features_batch5000(df):
    df['signal_batch_5k_min'] = df.groupby('batch_slices2')['signal'].transform('min')
    df['signal_batch_5k_max'] = df.groupby('batch_slices2')['signal'].transform('max')
    df['signal_batch_5k_std'] = df.groupby('batch_slices2')['signal'].transform('std')
    df['signal_batch_5k_mean'] = df.groupby('batch_slices2')['signal'].transform('mean')
    df['mean_abs_chg_batch_5k'] = df.groupby(['batch_slices2'])['signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))
    df['abs_max_batch_5k'] = df.groupby(['batch_slices2'])['signal'].transform(lambda x: np.max(np.abs(x)))
    df['abs_min_batch_5k'] = df.groupby(['batch_slices2'])['signal'].transform(lambda x: np.min(np.abs(x)))

    df['range_batch_5k'] = df['signal_batch_5k_max'] - df['signal_batch_5k_min']
    df['maxtomin_batch_5k'] = df['signal_batch_5k_max'] / df['signal_batch_5k_min']
    df['abs_avg_batch_5k'] = (df['abs_min_batch_5k'] + df['abs_max_batch_5k']) / 2
    return df
def set_shift_features(df):
    df['signal_shift+1'] = df.groupby(['batch']).shift(1)['signal']
    df['signal_shift-1'] = df.groupby(['batch']).shift(-1)['signal']
    df['signal_shift+2'] = df.groupby(['batch']).shift(2)['signal']
    df['signal_shift-2'] = df.groupby(['batch']).shift(-2)['signal']
    return df
def set_difference_features(df, ignore=['open_channels', 'time', 'batch', 'batch_index', 'batch_slices', 'batch_slices2',]):
    for c in list(set(df.columns) ^ set(ignore)):
        df[f'{c}_msignal'] = df[c] - df['signal']  
    return df
def set_gradients_features(df, n_grads=4):
    for i in range(n_grads):
        if i == 0:
            df['grad_' + str(i+1)] = df.groupby(['batch'])['signal'].transform(lambda x: np.gradient(x))
        else:
            df['grad_' + str(i+1)] = df.groupby(['batch'])['grad_' + str(i)].transform(lambda x: np.gradient(x))
    return df
def set_features(df, is_test=False, memory_reduce=True):
    print('set_index()')
    df = set_index(df)
    print('set_batch_index()')
    df = set_batch_index(df)
    print('set_features_batch50000()')
    df = set_features_batch50000(df)
    print('set_features_batch5000()')
    df = set_features_batch5000(df)
    print('set_lag_features()')
    df = set_shift_features(df)
    print('set_gradients_features()')
    df = set_gradients_features(df)
    
    print('set_difference_features()')
    if not is_test:
        df = set_difference_features(df, ignore=['open_channels', 'time', 'batch', 'batch_index', 'batch_slices', 'batch_slices2'])
    else:
        df = set_difference_features(df, ignore=['time', 'batch', 'batch_index', 'batch_slices', 'batch_slices2'])
    
    df = df.fillna(0)
    
    if memory_reduce:
        print('reduce_mem_usage()')
        df = reduce_mem_usage(df)
    return df
train_data = set_features(train_data)

pd.set_option('display.max_columns', 200)
train_data.head(10)
frac = 1.0
train_data = train_data.sample(frac=frac, random_state=42).reset_index(drop=True)
!pip install pycaret
IGNORE_FEATURES  = [
                    'time',
                    'batch',
                    'batch_index',
                    'batch_slices',
                    'batch_slices2',
                    'abs_max_batch',
                    'abs_min_batch',
                    'abs_avg_batch',
                    'signal_batch_min_msignal',
                    'signal_batch_mean_msignal',
                    'range_batch_5k_msignal'
                   ]

print('TARGET FEATURE LIST : ', end="")
print([f for f in list(set(IGNORE_FEATURES) ^ set(train_data.columns))])
from pycaret.regression import *
exp = setup(data = train_data, 
            target = 'open_channels',
            silent=True,
            sampling = False,
            ignore_features = IGNORE_FEATURES,
            session_id=42)
lgbm_model = create_model('lightgbm', fold=10)
lgbm_model = finalize_model(lgbm_model)
test_data = set_features(test_data, is_test=True)
test_data.head()
predictions = predict_model(lgbm_model, data=test_data)
predictions['open_channels'] = predictions['Label']
sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")

submission = pd.DataFrame()
submission['time']  = sub['time']
submission['open_channels'] = predictions['open_channels']
submission['open_channels'] = submission['open_channels'].round(decimals=0)
submission['open_channels'] = submission['open_channels'].astype(int)
submission.to_csv('submission.csv', float_format='%0.4f', index = False)