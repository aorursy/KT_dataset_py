# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing, metrics
from ipywidgets import widgets, interactive
import gc
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta 
from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm
from itertools import cycle
import datetime as dt
from torch.autograd import Variable
import random 
import os
from matplotlib.pyplot import figure
from fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time 
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import torch 

%matplotlib inline
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
def kernel_weight(test_X, train_X):
    kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=100).fit(test_X)
    d_test = np.exp(kde.score_samples(test_X))
    d_train = np.exp(kde.score_samples(train_X))
    return d_test / d_train
cat_feats = ['item_id', 'dept_id', 'cat_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
useless_cols = ["id", "date", "sales", "d"]
float_cols = [f'rolling_mean_{i}_momentum_{j}' for i in [7, 14, 30, 60] for j in [7, 14]] + [f'rolling_mean_{i}_momentum_{j}' for i in [7, 14, 30] for j in ['y', 'm']]
STORES = [f'CA_{i}' for i in range(1, 5)] + [f'TX_{i}' for i in range(1, 4)] + [f'WI_{i}'for i in range(1, 4)]
for STORE in STORES:
    start = time.time()
    print("Data loading {}".format(STORE))
    data_store = joblib.load(f'kaggle/input/data_06_28/data_{STORE}.pkl')
    train_cols = data_store.columns[~data_store.columns.isin(useless_cols)]
    weight_cols = data_store.columns[~data_store.columns.isin(useless_cols + cat_feats + float_cols)]
    # print(data_store[weight_cols].info())
    print(len(train_cols))
    # X_train.dropna(inplace=True)
    X_train = data_store[data_store.date <= '2016-05-22']
    y_train = X_train['sales']
    X_valid = data_store[(data_store.date >= '2016-04-25') & (data_store.date <= '2016-05-22')]
    y_valid = X_valid['sales']
    X_eval = data_store[data_store.date >= '2016-05-23']
    params = joblib.load(f'kaggle/input/model_06_30/model_{STORE}.pkl').get_params()
    test_X = X_eval[weight_cols].copy()
    train_X = X_train[weight_cols].copy()
    test_X.fillna(0, inplace=True)
    train_X.fillna(0, inplace=True)
    # train_X[float_cols] = train_X[float_cols].astype(np.float16)
    # test_X[float_cols] = test_X[float_cols].astype(np.float16)
    weight = kernel_weight(test_X=test_X, train_X=train_X)
    locals()[f'model_{STORE}'] = lgb.LGBMRegressor(**params_all, weight=weight)
    eval(f'model_{STORE}').fit(X_train[train_cols], y_train)
    end = time.time()
    print((end-start) / 60)
    joblib.dump(eval(f'model_{STORE}'), f'model_{STORE}.pkl')
    y_valid_pred = eval(f'model_{STORE}').predict(X_valid[train_cols])
    print(np.sqrt(metrics.mean_squared_error(y_valid_pred, y_valid)))
