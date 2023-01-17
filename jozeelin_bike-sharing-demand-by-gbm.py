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
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
train = pd.read_csv("../input/train.csv", parse_dates=[0],
                    date_parser=lambda d: pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
train['year'] = train['datetime'].map(lambda d: d.year)
train['month'] = train['datetime'].map(lambda d: d.month)
train['hour'] = train['datetime'].map(lambda d: d.hour)
train['weekday'] = train['datetime'].map(lambda d: d.weekday())
train['day'] = train['datetime'].map(lambda d: d.day)
train['weather'] = train['weather'].astype('category')
train['holiday'] = train['holiday'].astype('category')
train['workingday'] = train['workingday'].astype('category')
train['season'] = train['season'].astype('category')
train['hour'] = train['hour'].astype('category')
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
            'windspeed', 'hour', 'weekday', 'year']
booster = GradientBoostingRegressor(n_estimators=500,learning_rate=0.03,max_depth=10,min_samples_leaf=20)
booster_casual = booster.fit(train[features], np.log(train['casual']+1))
booster_reg = booster.fit(train[features], np.log(train['registered']+1))
test = pd.read_csv("../input/test.csv", parse_dates=[0],
                    date_parser=lambda d: pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
test['year'] = test['datetime'].map(lambda d: d.year)
test['month'] = test['datetime'].map(lambda d: d.month)
test['hour'] = test['datetime'].map(lambda d: d.hour)
test['weekday'] = test['datetime'].map(lambda d: d.weekday())
test['day'] = test['datetime'].map(lambda d: d.day)
test['weather'] = test['weather'].astype('category')
test['holiday'] = test['holiday'].astype('category')
test['workingday'] = test['workingday'].astype('category')
test['season'] = test['season'].astype('category')
test['hour'] = test['hour'].astype('category')
casual_pred = booster_casual.predict(test[features])
reg_pred = booster_reg.predict(test[features])
count = np.round(np.exp(casual_pred)-1+np.exp(reg_pred)-1)
df_result = pd.DataFrame({'datetime': test['datetime'], 'count': count})
df_result.to_csv('submission.csv', index = False)
