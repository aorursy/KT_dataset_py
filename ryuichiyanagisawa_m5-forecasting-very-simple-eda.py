import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 50)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import dask_xgboost as xgb

import dask.dataframe as dd

from sklearn import preprocessing, metrics

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reduce memory function

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

print('Reading files')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar = reduce_mem_usage(calendar)

print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sell_prices = reduce_mem_usage(sell_prices)

print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

print('Reading success')
sales_train_validation.head()
calendar.head()
sell_prices.head()
print(sales_train_validation.shape)

print(calendar.shape)

print(sell_prices.shape)

print("*"*50)





print(sales_train_validation.info())

print("*"*50)

print(calendar.info())

print("*"*50)

print(sell_prices.info())
#sales_train_validation missingdata

total = sales_train_validation.isnull().sum().sort_values(ascending=False)

percent = (sales_train_validation.isnull().sum()/sales_train_validation.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
#calendar missingdata

total = calendar.isnull().sum().sort_values(ascending=False)

percent = (calendar.isnull().sum()/calendar.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#calendar missingdata

total = sell_prices.isnull().sum().sort_values(ascending=False)

percent = (sell_prices.isnull().sum()/sell_prices.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
sales_train_validation.describe()
calendar.describe()
sell_prices.describe()
#target

sns.distplot(sell_prices['sell_price'])
numerical_feats_sales_train_validation = sales_train_validation.dtypes[sales_train_validation.dtypes != "object"].index

categorical_feats_sales_train_validation = sales_train_validation.dtypes[sales_train_validation.dtypes == "object"].index



numerical_feats_calendar = calendar.dtypes[calendar.dtypes != "object"].index

categorical_feats_calendar = calendar.dtypes[calendar.dtypes == "object"].index



numerical_feats_sell_prices = sell_prices.dtypes[sell_prices.dtypes != "object"].index

categorical_feats_sell_prices = sell_prices.dtypes[sell_prices.dtypes == "object"].index
for col in numerical_feats_sales_train_validation:

    print('{:15}'.format(col), 

          'Mean: {:05.2f}'.format(sales_train_validation[col].mean()) , 

          '   ' ,

          'Std: {:05.2f}'.format(sales_train_validation[col].std()) , 

          '   ' ,

          'Skewness: {:05.2f}'.format(sales_train_validation[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(sales_train_validation[col].kurt())  

         )

for col in numerical_feats_calendar:

    print('{:15}'.format(col), 

          'Mean: {:05.2f}'.format(calendar[col].mean()) , 

          '   ' ,

          'Std: {:05.2f}'.format(calendar[col].std()) , 

          '   ' ,

          'Skewness: {:05.2f}'.format(calendar[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(calendar[col].kurt())  

         )



for col in numerical_feats_sell_prices:

    print('{:15}'.format(col), 

          'Mean: {:05.2f}'.format(sell_prices[col].mean()) , 

          '   ' ,

          'Std: {:05.2f}'.format(sell_prices[col].std()) , 

          '   ' ,

          'Skewness: {:05.2f}'.format(sell_prices[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(sell_prices[col].kurt())  

         )





for catg in list(categorical_feats_sales_train_validation) :

    print(sales_train_validation[catg].value_counts())

    print('#'*50)
for catg in list(categorical_feats_calendar) :

    print(calendar[catg].value_counts())

    print('#'*50)
for catg in list(categorical_feats_sell_prices) :

    print(sell_prices[catg].value_counts())

    print('#'*50)
plt.figure(figsize=(18,9))

sns.heatmap(calendar.isnull(), cbar=False)
fig, ax = plt.subplots(figsize=(12, 9)) 

sns.heatmap(sales_train_validation.corr(), square=True, vmax=1, vmin=-1, center=0)
fig, ax = plt.subplots(figsize=(12, 9)) 

sns.heatmap(sell_prices.corr(), square=True, vmax=1, vmin=-1, center=0)
fig, ax = plt.subplots(figsize=(12, 9)) 

sns.heatmap(calendar.corr(), square=True, vmax=1, vmin=-1, center=0)
#train.corr()['target'].sort_values()

#train['target'].value_counts()