import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet # time series

import seaborn as sns # graphics

from scipy import stats # self-explanatory

import matplotlib.pyplot as plt

from matplotlib import rcParams

import itertools

import warnings

import statsmodels.api as sm



# global settings

warnings.filterwarnings("ignore")

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['text.color'] = 'k'

# figure size in inches

rcParams['figure.figsize'] = 11.7,8.27

plt.style.use('fivethirtyeight')

%matplotlib inline



# from fbprophet.plot import plot_plotly

# import plotly.offline as py

# py.init_notebook_mode()





import os

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# Load memory saving function (https://www.kaggle.com/ragnar123/very-fst-model)



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



calendar = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")

sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")

sales_train = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv")

samples_sub = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv")



calendar = reduce_mem_usage(calendar)

sell_prices = reduce_mem_usage(sell_prices)

sales_train = reduce_mem_usage(sales_train)

samples_sub = reduce_mem_usage(samples_sub)
calendar.info()
calendar.head()
print("Event Name 1:" + str(calendar['event_name_1'].unique()))

print('\n')

print("Event Type 1:" + str(calendar['event_type_1'].unique()))

print('\n')

print("Event Name 2:" + str(calendar['event_name_2'].unique()))

print('\n')

print("Event Type 2:" + str(calendar['event_type_2'].unique()))
sales_train.head()
sp_short = sales_train.iloc[:, np.r_[1,4, 6:1919]] 
sp_short.head()
sell_prices.head()
# runs out of memory here!!!! 



# items_and_prices=sp_short.merge(sell_prices, how='inner')
# There are several dimensions along which we may wish to aggregate our data; we do that here

# item_aggregate=sales_train.groupby('item_id').sum()

# dept_aggregate=sales_train.groupby('dept_id').sum()

# store_aggregate=sales_train.groupby('store_id').sum()

# state_aggregate=sales_train.groupby('state_id').sum()