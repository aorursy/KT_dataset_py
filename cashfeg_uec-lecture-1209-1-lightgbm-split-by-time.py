# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import lightgbm as lgb

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/ds2019uec-task1/train.csv")

test_df = pd.read_csv("../input/ds2019uec-task1/test.csv")
train = train_df.copy()

test = test_df.copy()
encoders = dict()

for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:

    le = preprocessing.LabelEncoder()

    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))

    train[col] = le.transform(train_df[col])

    test[col] = le.transform(test_df[col])

    encoders[col] = le
train["y"] = (train_df["y"] == "yes").astype(np.int)
period = np.arange(0, len(train)) // (len(train) // 4)

period = np.clip(period, 0, 3)
# check

period
param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'
va_period_list = [1, 2, 3]

for va_period in va_period_list:

    is_tr = period < va_period

    is_va = period == va_period

    tr_x, va_x = train[is_tr], train[is_va]

    tr_y, va_y = train["y"][is_tr], train["y"][is_va]

    

    train_data = lgb.Dataset(tr_x, tr_y)

    valid_data = lgb.Dataset(va_x, va_y)

    

    lgb.train(param, train_data, 50, valid_sets=[train_data, valid_data], verbose_eval=True)
train_x = train.drop(["y", "index", "duration", "month", "day"], axis=1)

train_y = train["y"]
va_period_list = [1, 2, 3]

for va_period in va_period_list:

    is_tr = period < va_period

    is_va = period == va_period

    tr_x, va_x = train_x[is_tr], train_x[is_va]

    tr_y, va_y = train_y[is_tr], train_y[is_va]

    

    train_data = lgb.Dataset(tr_x, tr_y)

    valid_data = lgb.Dataset(va_x, va_y)

    

    lgb.train(param, train_data, 50, valid_sets=[train_data, valid_data], verbose_eval=True)
train_all_data = lgb.Dataset(train_x, train_y)

bst = lgb.train(param, train_all_data, 20, valid_sets=[train_all_data], verbose_eval=True)
lgb.plot_importance(bst, figsize=(12, 6))
test_x = test.drop(["index", "duration", "month", "day"], axis=1)

y_pred = bst.predict(test_x)
sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = y_pred

sub.to_csv("sample_lgbm.csv", index=False)
train_y_pred = bst.predict(train_x)

plt.hist(train_y_pred)
plt.hist(y_pred)