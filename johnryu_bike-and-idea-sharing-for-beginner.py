# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load basic library
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')
train.head()
print(train.shape, test.shape)
train.info(), test.info()
train.describe()
# make a tempdate

train['tempdate'] = train.datetime.apply(lambda x : x.split())
train['tempdate'].head()
# Use apply function
train['year'] = train.tempdate.apply(lambda x : x[0].split('-')[0])
train['month'] = train.tempdate.apply(lambda x : x[0].split('-')[1])
train['day'] = train.tempdate.apply(lambda x : x[0].split('-')[2])
train['hour'] = train.tempdate.apply(lambda x : x[1].split(':')[0])
# check the year
train['year']
# transfer proper type
train['year'] = pd.to_numeric(train['year'])
train['month'] = pd.to_numeric(train['month'])
train['day'] = pd.to_numeric(train['day'])
train['hour'] = pd.to_numeric(train['hour'])
# When we think about the bike sharing. Day is not a important factor, but weekday is import factor.(I think so..)
train['weekday'] = train.tempdate.apply(lambda x : datetime.strptime(x[0],'%Y-%m-%d').weekday())
# Checking is important for beginner(it's make me comportable)
train['weekday']
# I don't need more tempdate. Drop in the castle!!!
train = train.drop(['tempdate'],axis=1)
figure = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,2,1)
ax1 = sns.barplot(x = 'season', y='count', data=train)

ax2 = plt.subplot(2,2,2)
ax2 = sns.barplot(x = 'holiday', y='count', data=train )

ax3 = plt.subplot(2,2,3)
ax3 = sns.barplot(x = 'workingday', y='count', data=train )

ax4 = plt.subplot(2,2,4)
ax4 = sns.barplot(x = 'weather', y='count', data=train )
figure = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,2,1)
ax1 = sns.barplot(x = 'year', y='count', data=train)

ax2 = plt.subplot(2,2,2)
ax2 = sns.barplot(x = 'month', y='count', data=train )

ax3 = plt.subplot(2,2,3)
ax3 = sns.barplot(x = 'day', y='count', data=train )

ax4 = plt.subplot(2,2,4)
ax4 = sns.barplot(x = 'hour', y='count', data=train )
# make function for change season

def seson_change(month):
    if month in [12,1,2]:
        return 4
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    elif month in [9,10,11]:
        return 3

train['season'] = train.month.apply(seson_change)
# One more visualization
figure = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,2,1)
ax1 = sns.barplot(x = 'season', y='count', data=train)

ax2 = plt.subplot(2,2,2)
ax2 = sns.barplot(x = 'holiday', y='count', data=train )

ax3 = plt.subplot(2,2,3)
ax3 = sns.barplot(x = 'workingday', y='count', data=train )

ax4 = plt.subplot(2,2,4)
ax4 = sns.barplot(x = 'weather', y='count', data=train )
figure = plt.figure(figsize = (10,10))

ax1 = plt.subplot(2,2,1)
ax1 = sns.distplot(train.temp)

ax2 = plt.subplot(2,2,2)
ax2 = sns.distplot(train.atemp)

ax3 = plt.subplot(2,2,3)
ax3 = sns.distplot(train.humidity)

ax4 = plt.subplot(2,2,4)
ax4 = sns.distplot(train.windspeed)
figure = plt.figure(figsize = (20,20))

ax = sns.heatmap(train.corr(), annot=True,vmin =0)
sns.jointplot(x='temp', y='atemp', data=train)
# Draw mix value graph
figure = plt.figure(figsize = (10,10))
list = ['season','holiday','weekday','weather']
i=0

for j in list:
    i = i+1
    plt.subplot(2,2,i)
    sns.pointplot(x='hour', y='count', hue=j,data=train)
fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot(2,1,1)
ax1 = sns.pointplot(x='month',y='count', hue='weather',data= train)

ax2 = plt.subplot(2,1,2)
ax2 = sns.barplot(x='month',y='count',data=train, hue='weather')
# before preprocessing we need to load the first train data

train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')
# concentrate the train, test
Total = pd.concat([train,test],axis=0)
Total
# Do the same process, once more for preprocessing

Total['tempdate'] = Total.datetime.apply(lambda x : x.split())

Total['year'] = Total.tempdate.apply(lambda x : x[0].split('-')[0])
Total['month'] = Total.tempdate.apply(lambda x : x[0].split('-')[1])
Total['day'] = Total.tempdate.apply(lambda x : x[0].split('-')[2])
Total['hour'] = Total.tempdate.apply(lambda x : x[1].split(':')[0])
Total['weekday'] = Total.tempdate.apply(lambda x : datetime.strptime(x[0], "%Y-%m-%d").weekday())

Total['year'] = pd.to_numeric(Total.year)
Total['month'] = pd.to_numeric(Total.month)
Total['day'] = pd.to_numeric(Total.day)
Total['hour'] = pd.to_numeric(Total.hour)

Total = Total.drop('tempdate',axis=1)

Total['season'] = Total.month.apply(seson_change)
# load forest
from sklearn.ensemble import RandomForestRegressor
# See the wind
sns.distplot(train['windspeed'])
# first, seperate wind '0'&'1' value
Total_wind_speed0 = Total[Total['windspeed'] == 0]
Total_wind_speed1 = Total[Total['windspeed'] != 0]

# When we make a wind in the forest, there are useless value in the train data
temp_drop_columns = ['windspeed','casual','registered','count','datetime','holiday','workingday']

# Save after drop columns
Total_wind_speed0_df = Total_wind_speed0.drop(temp_drop_columns, axis = 1)
Total_wind_speed1_df = Total_wind_speed1.drop(temp_drop_columns, axis = 1)

# Save the value for predict
Total_wind_speed1_df_speed = Total_wind_speed1['windspeed']
# simple train forest

rf = RandomForestRegressor()

rf.fit(Total_wind_speed1_df, Total_wind_speed1_df_speed)

# make wind using forest
Total_wind_speed0_df_speed = rf.predict(Total_wind_speed0_df)

# Save wind
Total_wind_speed0['windspeed'] = Total_wind_speed0_df_speed
# Remake the Total with wind
Total = pd.concat([Total_wind_speed0,Total_wind_speed1],axis=0)
# This columns have several values, but it's a category value. 
# I will divided several columns by using get dummies function

cat_columns = [ 'season', 'weather','year', 'hour', 'weekday' ] 

# Of coures, there are several category value that not contain cat_columns, but don't worry. i will drop the columns

for i in cat_columns:
    Total[i] = Total[i].astype('category')

# get_dummies
weather_df = pd.get_dummies(Total['weather'],prefix = 'w')
season_df = pd.get_dummies(Total['season'],prefix='s')
year_df = pd.get_dummies(Total['year'], prefix = 'y')
hour_df = pd.get_dummies(Total['hour'], prefix = 'h')
weekday_df = pd.get_dummies(Total['weekday'], prefix='we')
# Sum dummies data
Total = pd.concat([Total,weather_df], axis=1)
Total = pd.concat([Total,season_df], axis=1)
Total = pd.concat([Total,year_df], axis=1)
Total = pd.concat([Total,hour_df], axis=1)
Total = pd.concat([Total,weekday_df], axis=1)

# Data Checking is alway right 
Total.head()
Total.head()
# If you use log, '0' value tranfered infinity value(it make error)
# So, log1p make log(data +1) it's avoid error
Total['atemp'] = np.log1p(Total['atemp'])
Total['humidity'] = np.log1p(Total['humidity'])
Total['windspeed'] = np.log1p(Total['windspeed'])


#checking is alway right.
Total.head()
# divide values(seperate by using 'count'values)
train = Total[pd.notnull(Total['count'])]
test = Total[~pd.notnull(Total['count'])]

print(train.shape, test.shape)
drop_columns = ['casual', 'count','temp', 'datetime','registered','weather','season','year','month','day','hour','weekday']

# But, before we drop, we have to save some value(datetime need for submission, count need for prediction)

test_datetime = test['datetime']
y_label = train['count']

train = train.drop(drop_columns,axis=1)
test = test.drop(drop_columns,axis=1)
print(train.shape, test.shape)
print(y_label.shape)
# load library for predict
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
# save some model
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
# make label value transfer logvlaue for predict
y_label_log = np.log1p(y_label)

# train model
lr.fit(train,y_label_log)
rf.fit(train,y_label_log)
ridge.fit(train,y_label_log)
lasso.fit(train,y_label_log)

# predict easy
lr_preds = lr.predict(train)
rf_preds = rf.predict(train)
ridge_preds = ridge.predict(train)
lasso_preds = lasso.predict(train)
# we will use np.exp for transfer log value
# and just use MAE because it's easy
print("lr MAE :{:.5f}".format(metrics.mean_absolute_error(np.exp(y_label_log),np.exp(lr_preds))))
print("rf MAE :{:.5f}".format(metrics.mean_absolute_error(np.exp(y_label_log),np.exp(rf_preds))))
print("ridge MAE :{:.5f}".format(metrics.mean_absolute_error(np.exp(y_label_log),np.exp(ridge_preds))))
print("lasso MAE :{:.5f}".format(metrics.mean_absolute_error(np.exp(y_label_log),np.exp(lasso_preds))))
# I just borrowed idea from EDA & Ensemble Model (Top 10 Percentile) kernel
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb_params={'max_depth':range(1,11,1),'n_estimators':[1,10,100]}
acc_scorer = metrics.make_scorer(metrics.mean_absolute_error,greater_is_better=False)
grid_gb=GridSearchCV(gb,gb_params,scoring=acc_scorer,cv=5)
grid_gb.fit(train,y_label_log)
preds = grid_gb.predict(train)
print("GB MAE :{:.5f}".format(metrics.mean_absolute_error(np.exp(y_label_log),np.exp(preds))))
predsTest = grid_gb.predict(test)

# make dataFrame
submission = pd.DataFrame({
        "datetime": test_datetime,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })

# make file
submission.to_csv('bike_and_idea_sharing_for_beginner.csv', index=False)

