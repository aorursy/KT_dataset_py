# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/clubmahindra/train.csv')

test_df = pd.read_csv('/kaggle/input/club-mahindra-test/test.csv')

#submit = pd.read_csv('/kaggle/input/sample_submission_dlc0jkw/sample_submission.csv')

train_df.head(10)
print(train_df.shape)
train_df.describe()

train_df.info()
train_df.duplicated().sum()
# Identifying categorical and numerical columns

train_df.columns
# Converting to Date time # Use small letter for two digit year values otherwise python would give error

train_df['checkout_date'] = pd.to_datetime(train_df['checkout_date'],format = '%d/%m/%y')

train_df['checkin_date'] = pd.to_datetime(train_df['checkin_date'],format = '%d/%m/%y')
train_df['booking_date'] = pd.to_datetime(train_df['booking_date'],format = '%d/%m/%y')
train_df.head(10)
train_df['numberof_days_stayed'] = (train_df.checkout_date - train_df.checkin_date)
#train_df.head(10)

print(train_df.numberof_days_stayed.describe())

# roomnights column has a negative value which is not possible hence deleting that 

train_df[train_df.roomnights > -1]
# Feature engineering

# adding a feature 

## add columns relevant in a date column

## used from fastai

def add_datepart(df, fldname, drop=True, time=False):

    "Helper function that adds columns relevant to a date."

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)


## add date information columns for each datetype. Although all are not essential but add them anyway

add_datepart(train_df, 'booking_date')

add_datepart(train_df, 'checkin_date')

add_datepart(train_df, 'checkout_date')
## split the data to train and test set based on the type level

## define y (target) column

train = train_df

test = test_df

y = train['amount_spent_per_room_night_scaled']



## delete the  id columns which are not part of the model

train_id = train.reservation_id

test_id = test.reservation_id

#rain = train.drop(['type','amount_spent_per_room_night_scaled','reservation_id'], axis = 1)

#test = test.drop(['type','amount_spent_per_room_night_scaled','reservation_id'], axis = 1)


## split the dataset to train and valid.

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.25, random_state=2)
max(y), min(y)
## define rmse as per problem statement

def rmse(x,y): return math.sqrt(((x-y)**2).mean()) * 100
types(train)
# Trying Light GBM

from lightgbm import LGBMRegressor

lgb = LGBMRegressor(random_state=1, n_jobs=-1,n_estimators=300, num_leaves=15, learning_rate=0.1,min_child_samples=3, 

                    reg_alpha=0.0, reg_lambda=0.0, importance_type='gain')

lgb.fit(X_train, y_train)

y_valid_pred = lgb.predict(X_valid)

rmse(y_valid, y_valid_pred)