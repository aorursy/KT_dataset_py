%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

train=pd.read_csv('../input/train.csv')

train.head()
train.isnull().sum()
train.season.unique()
train.weather.value_counts()
train.holiday.value_counts()
sns.barplot(x='season', y='count', data=train)
sns.barplot(x='weather', y='count', data=train)
train[['count', 'holiday']].groupby(['holiday'], as_index = True).mean().sort_values(by = 'count')
train[['count', 'season']].groupby(['season'], as_index = True).mean().sort_values(by = 'count')
train["hour"] = [t.hour for t in pd.DatetimeIndex(train.datetime)]

train["day"] = [t.dayofweek for t in pd.DatetimeIndex(train.datetime)]

train["month"] = [t.month for t in pd.DatetimeIndex(train.datetime)]

train['year'] = [t.year for t in pd.DatetimeIndex(train.datetime)]

train['year'] = train['year'].map({2011:0, 2012:1})
X, y = train.iloc[:, 1:], train['count']
plt.scatter(x = train['casual'] + train['registered'], y = train['count'])

plt.show()
X = X.drop(['registered', 'casual', 'count'], axis=1)
from sklearn.cross_validation import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
from sklearn.preprocessing import StandardScaler

scl= StandardScaler()

X_train_std = scl.fit_transform(X_train)

X_test_std = scl.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 400, criterion='mse',random_state=1, n_jobs=-1)

forest.fit(X_train_std, y_train)

y_train_pred = forest.predict(X_train_std)

y_test_pred = forest.predict(X_test_std)
from sklearn.metrics import mean_squared_error, r2_score

#Root_Mean_Square_Log_Error(RMSE) is accuracy criteria for this problem

print('RMSLE train: %.3f' % np.sqrt(mean_squared_error(np.log(y_train + 1), np.log(y_train_pred + 1))))

print('RMSLE test: %.3f' % np.sqrt(mean_squared_error(np.log(y_test + 1), np.log(y_test_pred + 1))))

print('R2 train: %.3f' % r2_score(y_train, y_train_pred))

print('R2 test: %.3f' % r2_score(y_test, y_test_pred))
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()

clf.fit(X_train_std, y_train)

y_train_pred2 = clf.predict(X_train_std)

y_test_predd = clf.predict(X_test_std)

#Root_Mean_Square_Log_Error(RMSE) is accuracy criteria for this problem

print('RMSLE train: %.3f' % np.sqrt(mean_squared_error(np.log(y_train + 1), np.log(y_train_pred2 + 1))))

print('RMSLE test: %.3f' % np.sqrt(mean_squared_error(np.log(y_test + 1), np.log(y_test_predd + 1))))

print('R2 train: %.3f' % r2_score(y_train, y_train_pred2))

print('R2 test: %.3f' % r2_score(y_test, y_test_predd))
test=pd.read_csv('../input/test.csv')

test.head()
test["hour"] = [t.hour for t in pd.DatetimeIndex(test.datetime)]

test["day"] = [t.dayofweek for t in pd.DatetimeIndex(test.datetime)]

test["month"] = [t.month for t in pd.DatetimeIndex(test.datetime)]

test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]

test['year'] = test['year'].map({2011:0, 2012:1})
X_test=test.iloc[:,1:]
X_test = scl.transform(X_test)
y_test=forest.predict(X_test)
df_submit = test
df_submit['count'] = np.round(y_test)
df_submit = df_submit.drop(['season', 'holiday', 'workingday','weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour', 'day', 'month', 'year'], axis=1)
df_submit.head()
df_submit.to_csv('bike2.csv', index=False)