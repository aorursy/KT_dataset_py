# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from xgboost import XGBRegressor
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
submit = pd.read_csv('../input/sample_submission_V2.csv')
train.head()
test.head()
y_train = train['winPlacePerc']
x_train = train.drop(['Id','groupId','matchId','winPlacePerc', 'matchType'], axis=1)
x_test = test.drop(['Id','groupId','matchId','matchType'], axis=1)
x_train['matchDuration'] = x_train['matchDuration'].fillna(x_train['matchDuration'].mean())
x_test['matchDuration'] = x_test['matchDuration'].fillna(x_test['matchDuration'].mean())
x_train = x_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())
y_train = y_train.fillna(y_train.mean())
reg = XGBRegressor(n_estimators=1650, max_depth=9, learning_rate=0.007)
reg.fit(x_train, y_train, eval_set=[(x_train, y_train)])
reg.score(x_train, y_train)
output = pd.DataFrame()
output['Id'] = submit['Id']
y = reg.predict(x_test)
y[y<0]=0
output['winPlacePerc'] = y
output.to_csv('output.csv', index=False)
