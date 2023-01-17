import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import *

from sklearn.ensemble import *

from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV



os.chdir('D:\Kaggle\Sberbank Russian Housing Market\data')

macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)



# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130

ylog_train_all = np.log1p(df_train['price_doc'].values)

y_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')





# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp'], axis=1, inplace=True)



# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)

df_values.fillna(0)

df_values[df_values.isnull()]=0

# Convert to numpy values

x_all = df_values.values

x_train_all = x_all[:num_train]





X = Imputer().fit_transform(x_train_all)

Y = Imputer().fit_transform(ylog_train_all)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.