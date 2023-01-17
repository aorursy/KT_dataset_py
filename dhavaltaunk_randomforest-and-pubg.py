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
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submit = pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
train.info()
test.info()
y_train = train['winPlacePerc']
x_train = train.drop(['Id','groupId','matchId','winPlacePerc'], axis=1)
x_test = test.drop(['Id','groupId','matchId'], axis=1)
reg = RandomForestRegressor(n_estimators=128, max_depth=9 )
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
output = pd.DataFrame()
output['Id'] = submit['Id']
output['winPlacePerc'] = reg.predict(x_test)
output.to_csv('output.csv', index=False)
reg.score(x_test, submit['winPlacePerc'])
reg.predict(x_test)
