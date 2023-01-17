import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



import matplotlib.pyplot as plt # matplotlib and seaborn for plotting

import matplotlib.patches as patches

import seaborn as sns



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 150)

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import os,random, math, psutil, pickle



from time import time

import datetime

pd.set_option('display.max_columns',100)

pd.set_option('display.float_format', lambda x: '%.5f' % x)



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split,KFold

from sklearn import metrics

from sklearn.metrics import mean_squared_error

import lightgbm as lgb



root = '../input/ashrae-energy-prediction'
train = pd.read_csv(root + "/train.csv", parse_dates=['timestamp'])



weather_train = pd.read_csv(root+"/weather_train.csv",parse_dates=['timestamp'])



test_cols_to_read = ['building_id','meter','timestamp']

test = pd.read_csv(root+"/test.csv",parse_dates=['timestamp'],usecols=test_cols_to_read)



weather_test = pd.read_csv(root + "/weather_test.csv", parse_dates=['timestamp'])



building_meta = pd.read_csv(root + "/building_metadata.csv")



sample_submission = pd.read_csv(root + "/sample_submission.csv")
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
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)

weather_test = reduce_mem_usage(weather_test)

building_meta = reduce_mem_usage(building_meta)
print('train info',train.info())

print('-------------------')

print('weather_train info', weather_train.info())

print('-------------------')

print('test info', test.info()) 

print('-------------------')

print('weather_test info', weather_test.info())

print('-------------------')

print('building info', building_meta.info())