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
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
y_train = df_train['trip_duration']

df_train.drop(['id','trip_duration','store_and_fwd_flag', 'dropoff_datetime', 'pickup_datetime'], axis=1, inplace=True)

test_ids = df_test['id']

df_test.drop(['id', 'store_and_fwd_flag', 'pickup_datetime'], axis=1, inplace=True)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'reg:linear'

params['eval_metric'] = 'rmse'

params['eta'] = 0.02

params['max_depth'] = 5



d_train = xgb.DMatrix(df_train, label=y_train)



bst = xgb.train(params, d_train, 400, verbose_eval=10)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = test_ids

sub['trip_duration'] = p_test

sub.to_csv('simple_xgb.csv', index=False)