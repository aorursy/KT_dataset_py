import numpy as np

import pandas as pd

import os

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

import copy

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error, f1_score

pd.options.display.precision = 15

from collections import defaultdict

import lightgbm as lgb

import xgboost as xgb

import catboost as cat

import time

from collections import Counter

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

# import eli5

import shap

from IPython.display import HTML

import json

import altair as alt

from category_encoders.ordinal import OrdinalEncoder

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

from typing import List



import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

from typing import Any

from itertools import product

pd.set_option('max_rows', 500)

import re

from tqdm import tqdm

from joblib import Parallel, delayed
path = '/kaggle/input/m5-forecasting-accuracy'

train_sales = pd.read_csv(f'{path}/sales_train_validation.csv')

calendar = pd.read_csv(f'{path}/calendar.csv')

submission = pd.read_csv(f'{path}/sample_submission.csv')

sell_prices = pd.read_csv(f'{path}/sell_prices.csv')
calendar
train_sales
sell_prices
submission
train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_001']
sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_1_001']
plt.figure(figsize=(12, 4))

for i in range(10):

    plt.plot(train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].values,

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5]);

plt.title('HOBBIES_1_002 sales')

plt.legend();
plt.figure(figsize=(12, 4))

for i in range(10):

    plt.plot(train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(30).mean().values,

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5]);

plt.title('HOBBIES_1_002 sales, rolling mean 30 days')

plt.legend();



plt.figure(figsize=(12, 4))

for i in range(10):

    plt.plot(train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(60).mean().values,

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5]);

plt.title('HOBBIES_1_002 sales, rolling mean 60 days')

plt.legend();



plt.figure(figsize=(12, 4))

for i in range(10):

    plt.plot(train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 6:].rolling(90).mean().values,

             label=train_sales.loc[train_sales['item_id'] == 'HOBBIES_1_002'].iloc[i, 5]);

plt.title('HOBBIES_1_002 sales, rolling mean 90 days')

plt.legend();

item_prices = sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_2_001']

for s in item_prices['store_id'].unique():

    small_df = item_prices.loc[item_prices['store_id'] == s]

    plt.plot(small_df['wm_yr_wk'], small_df['sell_price'], label=s)

plt.legend()

plt.title('HOBBIES_2_001 sell prices');
train_sales.loc[train_sales['store_id'] == 'CA_1']
sell_prices.loc[sell_prices['store_id'] == 'CA_1']
ca_1_sales = train_sales.loc[train_sales['store_id'] == 'CA_1']

pd.crosstab(ca_1_sales['cat_id'], ca_1_sales['dept_id'])
plt.figure(figsize=(12, 4))

for d in ca_1_sales['dept_id'].unique():

    store_sales = ca_1_sales.loc[ca_1_sales['dept_id'] == d]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label=d)

plt.title('CA_1 sales by department, rolling mean 30 days')

plt.legend(loc=(1.0, 0.5));
item_prices = sell_prices.loc[sell_prices['item_id'] == 'HOBBIES_2_001']

for s in item_prices['store_id'].unique():

    small_df = item_prices.loc[item_prices['store_id'] == s]

    plt.plot(small_df['wm_yr_wk'], small_df['sell_price'], label=s)

plt.legend()

plt.title('HOBBIES_2_001 sell prices');
ca_1_prices = sell_prices.loc[sell_prices['store_id'] == 'CA_1']

ca_1_prices['dept_id'] = ca_1_prices['item_id'].apply(lambda x: x[:-4])

plt.figure(figsize=(12, 6))

for d in ca_1_prices['dept_id'].unique():

    small_df = ca_1_prices.loc[ca_1_prices['dept_id'] == d]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=d)

plt.legend(loc=(1.0, 0.5))

plt.title('CA_1 mean sell prices by dept');
train_sales.loc[train_sales['dept_id'] == 'HOBBIES_1']
train_sales.loc[train_sales['dept_id'] == 'HOBBIES_1', 'item_id'].nunique()
sell_prices.loc[sell_prices['item_id'].str.contains('HOBBIES_1')]
hobbies_1_sales = train_sales.loc[train_sales['dept_id'] == 'HOBBIES_1']

plt.figure(figsize=(12, 6))

for d in hobbies_1_sales['store_id'].unique():

    store_sales = hobbies_1_sales.loc[hobbies_1_sales['store_id'] == d]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label=d)

plt.title('HOBBIES_1 sales by stores, rolling mean 30 days')

plt.legend(loc=(1.0, 0.5));
sell_prices.head()
hobbies_1_prices = sell_prices.loc[sell_prices['item_id'].str.contains('HOBBIES_1')]

plt.figure(figsize=(12, 6))

for d in hobbies_1_prices['store_id'].unique():

    small_df = hobbies_1_prices.loc[hobbies_1_prices['store_id'] == d]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=d)

plt.legend(loc=(1.0, 0.5))

plt.title('HOBBIES_1 mean sell prices by store');
train_sales.loc[train_sales['state_id'] == 'CA']
for col in ['item_id', 'dept_id', 'store_id']:

    print(f"{col} has {train_sales.loc[train_sales['state_id'] == 'CA', col].nunique()} unique values for CA state")
ca_sales = train_sales.loc[train_sales['state_id'] == 'CA']

plt.figure(figsize=(12, 6))

for d in ca_sales['store_id'].unique():

    store_sales = ca_sales.loc[ca_sales['store_id'] == d]

    store_sales.iloc[:, 6:].sum().rolling(30).mean().plot(label=d)

plt.title('CA sales by stores, rolling mean 30 days')

plt.legend(loc=(1.0, 0.5));
ca_prices = sell_prices.loc[sell_prices['store_id'].str.contains('CA')]

plt.figure(figsize=(12, 6))

for d in ca_prices['store_id'].unique():

    small_df = ca_prices.loc[ca_prices['store_id'] == d]

    grouped = small_df.groupby(['wm_yr_wk'])['sell_price'].mean()

    plt.plot(grouped.index, grouped.values, label=d)

plt.legend(loc=(1.0, 0.5))

plt.title('Mean sell prices by store in CA');
train_sales.head()
plt.figure(figsize=(12, 8))

dept_grouped_sales = train_sales.groupby(['dept_id']).sum()

for i, row in dept_grouped_sales.iterrows():

    plt.plot(row.values, label=i);

plt.legend(loc=(1.0, 0.5))

plt.title('Sales by departments');
plt.figure(figsize=(12, 4))

for i, row in dept_grouped_sales.iterrows():

    plt.plot(row.rolling(30).mean().values, label=i);

plt.title('Sales by department, rolling mean 30 days')

plt.legend(loc=(1.0, 0.5));



plt.figure(figsize=(12, 4))

for i, row in dept_grouped_sales.iterrows():

    plt.plot(row.rolling(60).mean().values, label=i);

plt.title('Sales by department, rolling mean 60 days')

plt.legend(loc=(1.0, 0.5));



plt.figure(figsize=(12, 4))

for i, row in dept_grouped_sales.iterrows():

    plt.plot(row.rolling(90).mean().values, label=i);

plt.title('Sales by department, rolling mean 90 days')

plt.legend(loc=(1.0, 0.5));
