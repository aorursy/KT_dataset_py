# IMPORT LIBRARIES 

import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from catboost.utils import eval_metric
from catboost import CatBoostRegressor, Pool
import xgboost as xgb

import warnings
warnings.filterwarnings(action='ignore')
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
# LOAD TRAIN AND TEST DATA

df_train = pd.read_csv(r'/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv(r'/kaggle/input/liverpool-ion-switching/test.csv')
df_train = reduce_mem_usage(df_train)

print(df_train.head(3))
df_train.time.min(), df_train.time.max()
print(df_test.head(3))
df_test.time.min(), df_test.time.max()
# TRAIN AND TEST SIGNAL VISUALIZATION

df1 = df_train.copy()
df2 = df_test.copy()
df1['type'] = 'train'
df2['type'] = 'test'
df = df1
df = df.append(df2)

del df1
del df2


fig = px.line(df.iloc[::20], x="time", y="signal", color='type')
fig.show()
del df
# TRAIN AND TEST DATA SIGNAL & NUMBER OF OPEN CHANNELS VISUALIZATION


split_val = 50
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_train.iloc[::split_val]['time'], y=df_train.iloc[::split_val]['signal'],
                    mode='lines',
                    name='train'))
fig.add_trace(go.Scatter(x=df_test.iloc[::split_val]['time'], y=df_test.iloc[::split_val]['signal'],
                    mode='lines',
                    name='test'))

fig.add_trace(go.Scatter(x=df_train.iloc[::split_val]['time'], y=df_train.iloc[::split_val]['open_channels'],
                    mode='lines',
                    name='open_channels'))

fig.show()
df_train.head(3)
%%time

window_sizes = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]



def feature_engineering(df_train):
    for window in window_sizes:
        df_train['signal_' + "rolling_mean_" + str(window)] = df_train['signal'].rolling(window=window).mean()
        df_train['signal_' + "rolling_std_" + str(window)] = df_train['signal'].rolling(window=window).std()
        df_train['signal_' + "rolling_var_" + str(window)] = df_train['signal'].rolling(window=window).var()
        df_train['signal_' + "rolling_min_" + str(window)] = df_train['signal'].rolling(window=window).min()
        df_train['signal_' + "rolling_max_" + str(window)] = df_train['signal'].rolling(window=window).max()
        df_train['signal_' + "rolling_median_" + str(window)] = df_train['signal'].rolling(window=window).median()
        df_train['signal_' + "rolling_range_" + str(window)] = abs(df_train['signal_' + "rolling_max_" + str(window)] - df_train['signal_' + "rolling_min_" + str(window)])
        # adding covariance
        df_train['signal_' + "rolling_cov_" + str(window)] = df_train['signal'].rolling(window=window).cov()
        # adding skewnes - not working
#         df_train['signal_' + "rolling_skew_" + str(window)] = df_train['signal'].rolling(window=window).skew()
        # adding kurtosis - not working
#         df_train['signal_' + "rolling_kurt_" + str(window)] = df_train['signal'].rolling(window=window).kurt()

        # exponentially weighted parameters
        df_train['signal_' + "rolling_EW_mean_" + str(window)] = df_train['signal'].ewm(span=window).mean()
        df_train['signal_' + "rolling_EW_var_" + str(window)] = df_train['signal'].ewm(span=window).var()
        df_train['signal_' + "rolling_EW_std_" + str(window)] = df_train['signal'].ewm(span=window).std()
        df_train['signal_' + "rolling_EW_cov_" + str(window)] = df_train['signal'].ewm(span=window).cov()    
        
        # max2min
        df_train['signal_' + "rolling_max2min_" + str(window)] = abs(df_train['signal_' + "rolling_max_" + str(window)] / df_train['signal_' + "rolling_min_" + str(window)])
        # average max_min
        df_train['signal_' + "rolling_abs_avg_" + str(window)] = abs((df_train['signal_' + "rolling_max_" + str(window)] + df_train['signal_' + "rolling_min_" + str(window)])) / 2
        
    # lets add some lag for statistics
    signal_cols = [x for x in df_train.columns.tolist() if 'signal' in x]
    
    # add lags for data
    df_train['signal' + '_lagged_1minus'] = df_train['signal'].shift(-1)
    df_train['signal' + '_lagged_1plus'] = df_train['signal'].shift(1)
    df_train['signal' + '_lagged_2minus'] = df_train['signal'].shift(-2)
    df_train['signal' + '_lagged_2plus'] = df_train['signal'].shift(2)
    
        
    df_train = df_train.replace([np.inf, -np.inf], np.nan)    
    
    return df_train

df_train = feature_engineering(df_train)
df_train = reduce_mem_usage(df_train)
sorted(df_train.columns.tolist())
len(df_train.columns)
# lets get only required column which will participate in training and testing 
col = df_train.columns.tolist()
unwanted_num = {'time', 'open_channels'}
col = [ele for ele in col if ele not in unwanted_num] 
x1, x2, y1, y2 = train_test_split(df_train[col], df_train['open_channels'], test_size=0.3, random_state=7)
del df_train
model = CatBoostRegressor(random_seed=42, logging_level='Silent', iterations=700)

model.fit(
    x1, y1,
    eval_set=(x2, y2),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True, use_best_model=True)
model.get_feature_importance()
del x1
del x2
del y1
del y2

df_test = feature_engineering(df_test)

preds = model.predict(df_test[col])

df_test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)
df_test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')
df_test.head()
