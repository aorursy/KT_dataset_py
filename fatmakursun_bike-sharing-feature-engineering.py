# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.isnull().sum().sort_values(ascending = False)) #as you can see there is no null value in the columns

print("**"*50)

print(test.isnull().sum().sort_values(ascending = False))
print(train.info())

print('**'*50)

print(test.info())
train.datetime = pd.to_datetime(train.datetime)

test.datetime = pd.to_datetime(test.datetime)
print(train.info())

print('**'*50)

print(test.info())
train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['dayofweek'] = train['datetime'].dt.weekday_name





test['year'] = test['datetime'].dt.year

test['month'] = test['datetime'].dt.month

test['day'] = test['datetime'].dt.day

test['hour'] = test['datetime'].dt.hour

test['dayofweek'] = test['datetime'].dt.weekday_name
train.tail()
plt.figure(figsize=(16,8))

sns.heatmap(train.corr(), annot=True)

plt.show()
plt.figure(figsize=(16,8))

sns.distplot(train['count'])

plt.show()
plt.figure(figsize=(16,8))

plt.plot(train["count"][0:300])

plt.show()
# we need to convert categorical data to numeric data.



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['dayofweek'] = le.fit_transform(train['dayofweek'])

test['dayofweek'] = le.transform(test['dayofweek'])
plt.figure(figsize=(16,8))

sns.boxplot(x='season', y='count', data=train)
plt.figure(figsize=(16,8))

sns.boxplot(x='dayofweek',y='count', data=train)
plt.figure(figsize=(16,8))

sns.boxplot(x='hour',y='count', data=train)
plt.figure(figsize=(16,8))

sns.boxplot(x='year',y='count', data=train)
plt.figure(figsize=(16,8))

plt.hist(train['count'][train['year'] == 2011], alpha=0.5, label='2011')

plt.hist(train['count'][train['year'] == 2012], alpha=0.5, label='2012', color='red')
plt.scatter(train['hour'], train['count'])
train.head()
train.set_index('datetime', inplace=True)
train['2011-01-19 23:00:00':]
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train_without_outliers =train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
train_without_outliers.dropna(inplace=True)
print(train.info())

print('*********************************************************************************')

print(train_without_outliers.info())
train_without_outliers.head(2)
plt.figure(figsize=(12, 7))

sns.boxplot(x='season',y='windspeed',data=train_without_outliers,palette='winter')
def wind(cols):

    windspeed = cols[0]

    season = cols[1]

    

    if windspeed==0:



        if season == 1 :

            return 14



        elif season == 2 :

            return 14



        else:

            return 13



    else:

        return windspeed
train_without_outliers['wind'] = train_without_outliers[['windspeed','season']].apply(wind,axis=1)

test['wind'] = test[['windspeed', 'season']].apply(wind, axis=1)
test.head()
train_without_outliers.head(5)
train_without_outliers[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']] = train_without_outliers[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']].astype('category')

test[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']] = test[['season','holiday','workingday','weather', 'year','month','day','hour','dayofweek']].astype('category')
train_without_outliers.info()
from sklearn.model_selection import train_test_split
X = train_without_outliers[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','wind']]

y = train_without_outliers['count']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
y_train
y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

sc_y = MinMaxScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
from sklearn.metrics import mean_squared_error

from sklearn import metrics

print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))
plt.scatter(y_test,rf_prediction)
plt.figure(figsize=(16,8))

plt.plot(rf_prediction[0:200],'r')

plt.plot(y_test[0:200])
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()

dt_reg.fit(X_train, y_train)
dt_prediction = dt_reg.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, dt_prediction))
plt.scatter(y_test,dt_prediction)
test.head()
test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','wind']] = sc_X.fit_transform(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','wind']])
test_pred= rf.predict(test[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'year', 'month', 'day', 'hour', 'dayofweek','wind']])
test_pred
test_pred=test_pred.reshape(-1,1)
test_pred = sc_y.inverse_transform(test_pred)
test_pred = pd.DataFrame(test_pred, columns=['count'])
df = pd.concat([test['datetime'], test_pred],axis=1)
df.head()
df['count'] = df['count'].astype('int')
df.to_csv('submission1.csv' , index=False)