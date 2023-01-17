# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/bike-sharing-demand/train.csv')

test = pd.read_csv('../input/bike-sharing-demand/test.csv')
train.info()
train.head()
test.info()
tt = train.append(test)

tt.shape
tt.head()
tt.reset_index().drop('index', axis=1, inplace=True)

tt.head()
sns.distplot(train['count'])
pd.DatetimeIndex(train['datetime'])
# 时间处理

# 增加两列，分别为日期和小时

temp = pd.DatetimeIndex(train['datetime'])

train['year'] = temp.year

train['date'] = temp.date

train['hour'] = temp.hour
pd.DatetimeIndex(train.date).dayofweek
# 添加一个星期几的列

train['dayofweek'] = pd.DatetimeIndex(train.date).dayofweek
train.head()
# 一天中各时间段对count的影响

sns.boxplot(train['hour'], train['count'])
# 一周中各天对count的影响

sns.boxplot(train['dayofweek'], train['count'])
# 一周中每天count的变化

sns.pointplot(x='hour', y='count', hue='dayofweek', data=train)
pd.to_datetime(train['datetime'])
# 不同月份对count的影响

train['month'] = pd.to_datetime(train['datetime']).dt.month

sns.boxplot(train['month'], train['count'])
# 节假日对count的影响

sns.pointplot(x='hour', y='count', hue='workingday', data=train)
#天气对count的影响

plt.figure()

sns.boxplot(train['weather'], train['count'])

plt.figure()

sns.pointplot(x='hour', y='count', hue='weather', data=train)
# 季节对count的影响

plt.figure()

sns.boxplot(train['season'], train['count'])

plt.figure()

sns.pointplot(x='hour', y='count', hue='season', data=train)
# 皮尔逊系数

cor = train[['temp', 'atemp', 'casual', 'registered', 'humidity', 'windspeed', 'count']].corr()

sns.heatmap(cor, square=True, annot=True)
temp = pd.DatetimeIndex(tt['datetime'])

tt['year'] = temp.year

tt['hour'] = temp.hour

tt = tt[['hour', 'year', 'workingday', 'holiday', 'season', 'weather', 'atemp', 'count']]



# 对离散型变量做 one-hot 编码

tt = pd.get_dummies(tt, columns=['hour'], prefix=['hour'], drop_first=True)

tt = pd.get_dummies(tt, columns=['year'], prefix=['year'], drop_first=True)

tt = pd.get_dummies(tt, columns=['season'], prefix=['season'], drop_first=True)

tt = pd.get_dummies(tt, columns=['weather'], prefix=['weather'], drop_first=True)

tt.head()
new_train = tt.iloc[:10886, :]

new_test = tt.iloc[10886:, :].drop('count', axis=1)

# 因原count不符合正态分布

# 对 count + 1 然后取对数

y = np.log1p(new_train['count'])

new_train.drop('count', axis=1, inplace=True)

x = new_train

x.head()
new_test.shape, test.shape, train.shape, new_train.shape, tt.shape
y
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

gbr = GradientBoostingRegressor(random_state=3)
# 调参

params = {'learning_rate': [0.1, 0.01, 0.001, 0.5, 0.05],

          'n_estimators': [100, 200, 300, 400]}

grid = GridSearchCV(gbr, param_grid=params, cv=5)

grid.fit(x, y)
grid.best_params_
gbr = GradientBoostingRegressor(learning_rate=0.5, n_estimators=400)

cross_val_score(gbr, x, y, cv=5).mean()
gbr.fit(X_train, y_train)

pre = gbr.predict(X_test)

mean_squared_error(pre, y_test)
from sklearn.ensemble import RandomForestRegressor

rbf = RandomForestRegressor(n_estimators=222, random_state=50, max_features='sqrt')
cross_val_score(rbf, x, y, cv=5).mean()
gbr.fit(x, y)

co = gbr.predict(new_test)

m = []

for i in (np.exp(co) - 1):

    n = round(i)

    m.append(n)



predict = pd.DataFrame({'datetime': test['datetime'], 'count': m})

predict.to_csv('gbr.csv', index=False)