import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Read data

train = pd.read_csv('../input/train.csv' , parse_dates=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=['date'])
# Glimse data
print(train.shape)
train.head()
print(test.shape)
test.head()
# Combine train and test set
train['train_or_test'] = 'train'
test['train_or_test'] = 'test'

df = pd.concat([train,test] , axis = 0 , sort=False)
df.drop('sales' , axis = 1,inplace=True)
print('Combine df shape {}'.format(df.shape))
print(df.head())
df['store'].describe()
sns.distplot(df['store'], hist = True , kde = True)
plt.show()

sns.distplot(df['item'], hist=True)
df['item'].describe()

#Feature engineering
#Extracting day , month , year from colum date 
import time
df['dayofmonth'] = df.date.dt.day
df['dayofweek'] = df.date.dt.dayofweek
df['dayofyear'] = df.date.dt.dayofyear
df['month'] = df.date.dt.month
df['year'] = df.date.dt.year
df['weekofyear'] = df.date.dt.weekofyear
df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df.head()
from sklearn.ensemble import RandomForestRegressor

X_train = df[df['train_or_test'] == 'train'].drop(['date','id','train_or_test'], axis = 1)
Y_train = train['sales']

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, Y_train)

X_test = df[df['train_or_test'] == 'test'].drop(['date','id', 'train_or_test'] , axis = 1)
predicts = regr.predict(X_test)

subs = pd.DataFrame({'id' : test['id'] , 'sales' : predicts})
subs.to_csv('random_forest_predict.csv' , index = False)
subs.head()
