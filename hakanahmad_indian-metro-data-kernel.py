# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/Train.csv')

test = pd.read_csv('/kaggle/input/Test.csv')
train.head()
train.isnull().any()
train.dtypes
train.date_time = pd.to_datetime(train.date_time)
train['year'] = train['date_time'].dt.year

train['month'] = train['date_time'].dt.month

train['day'] = train['date_time'].dt.day

train['dayofweek'] = train['date_time'].dt.dayofweek.replace([0,1,2,3,4,5,6],['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])

train['hour'] = train['date_time'].dt.hour

train.head()
plt.figure(figsize=(10,7))

sns.lineplot(x=train['hour'],y=train['traffic_volume'])
plt.figure(figsize=(10,7))

sns.lineplot(x=train['hour'],y=train['traffic_volume'],hue=train['dayofweek'])
plt.figure(figsize=(10,7))

sns.lineplot(x=train['hour'],y=train['traffic_volume'],hue=train.query("dayofweek in ['monday','sunday']")['dayofweek'])
plt.figure(figsize=(10,7))

sns.lineplot(x=train['day'],y=train['traffic_volume'])
plt.figure(figsize=(10,7))

sns.lineplot(x=train['month'],y=train['traffic_volume'])
plt.figure(figsize=(10,7))

sns.lineplot(x=train['year'],y=train['traffic_volume'])
train.groupby('is_holiday').agg(len)['date_time'].plot.bar()
train.groupby('weather_type').agg(len)['date_time'].sort_values(ascending=False).plot.bar()
sns.pairplot(train[['air_pollution_index','humidity','wind_speed','visibility_in_miles','dew_point','temperature','rain_p_h','snow_p_h','clouds_all','traffic_volume']])
train['rain_p_h'] = train['rain_p_h'].replace(train['rain_p_h'].max(),train['rain_p_h'].median())

sns.distplot(a=train['rain_p_h'])
train['temperature'] = train['temperature'].replace(train['temperature'].min(),train['temperature'].median())

sns.distplot(a=train['temperature'])
X=train.drop(['date_time','is_holiday','weather_description','year','month','day','traffic_volume','dew_point'],axis=1)

y=train['traffic_volume']

X=pd.get_dummies(X)
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import mean_absolute_error



linreg = LinearRegression()

linreg.fit(X_train,y_train)

print('Cross Validation Score: ',-1*cross_val_score(linreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())
grafik_1 = pd.DataFrame({'Coef':linreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()

grafik_2 = pd.DataFrame({'Coef':linreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()

grafik = pd.concat([grafik_1,grafik_2])

grafik.plot.bar()
ridreg = Ridge()

ridreg.fit(X_train,y_train)

print('Cross Validation Score: ',-1*cross_val_score(ridreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())
grafik_1 = pd.DataFrame({'Coef':ridreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()

grafik_2 = pd.DataFrame({'Coef':ridreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()

grafik = pd.concat([grafik_1,grafik_2])

grafik.plot.bar()
lasreg = Lasso()

lasreg.fit(X_train,y_train)

print('Cross Validation Score: ',-1*cross_val_score(lasreg,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())
grafik_1 = pd.DataFrame({'Coef':lasreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).head()

grafik_2 = pd.DataFrame({'Coef':lasreg.coef_},index=X_train.columns).sort_values(by='Coef',ascending=False).tail()

grafik = pd.concat([grafik_1,grafik_2])

grafik.plot.bar()
from xgboost import XGBRegressor

warnings.simplefilter("ignore",UserWarning)

xgb = XGBRegressor(objective='reg:squarederror')

xgb.fit(X_train,y_train)

print('Cross Validation Score: ',-1*cross_val_score(xgb,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor()

rf.fit(X_train,y_train)

print('Cross Validation Score: ',-1*cross_val_score(rf,X_train,y_train,cv=5,scoring='neg_mean_absolute_error').mean())
hasil=[]

j=[]

for i in range(10,310,10):

    rf = RandomForestRegressor(n_estimators=i)

    rf.fit(X_train,y_train)

    pred = rf.predict(X_valid)

    hasil.append(mean_absolute_error(pred,y_valid))

    j.append(i)

score = pd.DataFrame({'Mean Absolute Error':hasil},index=j)

score.plot.line()
rf = RandomForestRegressor(n_estimators=150)

rf.fit(X_train,y_train)

pred = rf.predict(X_valid)

print('MAE: ',mean_absolute_error(pred,y_valid))
tabel = pd.DataFrame({'Importance':np.round(rf.feature_importances_,decimals=3)},index=X_train.columns).sort_values(by='Importance',ascending=False).head(10)

tabel
X=train[['hour','dayofweek','temperature','air_pollution_index','wind_direction']]

y=train['traffic_volume']

X=pd.get_dummies(X)

X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3,random_state=1)

rf=RandomForestRegressor(n_estimators=150)

rf.fit(X_train,y_train)

pred = rf.predict(X_valid)

print(mean_absolute_error(pred,y_valid))
test_1 = test.copy()

test_1['date_time'] = pd.to_datetime(test_1['date_time'])

test_1['dayofweek'] = test_1['date_time'].dt.dayofweek.replace([0,1,2,3,4,5,6],['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])

test_1['hour'] = test_1['date_time'].dt.hour

test_1 = test_1[['hour','dayofweek','temperature','air_pollution_index','wind_direction']]

test_1 = pd.get_dummies(test_1)
test_1.head()
final_pred = rf.predict(test_1)

final_test = pd.read_csv('/kaggle/input/Test.csv')

final_pred = pd.DataFrame(np.round(final_pred,decimals=0),columns=['Predictions'])

result = pd.concat([final_test,final_pred],axis=1)

result.head()
result.to_csv('result.csv')