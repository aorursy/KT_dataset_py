# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import calendar
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read data set
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
"""
datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals
"""

print('Attrs of train_dataset are {}'.format(train.columns))
print('Attrs of test_dataset are {}'.format(test.columns))
#Looking the shape of the train dataset
train.head()
#checking the null value of train_dataset
train.info()
#the range of datatime data set
print('the first datetime of dataset is {}, the last datetime of dataset is {}'.format(train.datetime.min(),train.datetime.max()))
#let's check the empty time of our dataset
#from 2011-01-01 to 2012-12-19, divided days into each hours
df_daterange = pd.DataFrame(pd.date_range("2011-01-01","2012-12-19",freq="H")).iloc[:-1] 
#print(pd.DataFrame(pd.date_range("2011-01-01","2012-12-19",freq="H")).iloc[:-1]) => 17232 rows
#10886 != 17232
train['date'] = train.datetime.apply(lambda x:x.split()[0])
train['month'] = train.date.apply(lambda dateString: dateString.split("-")[1])

#We need to clean our data correctly because Jan 1st is not in the Spring
def monthToSeason(month):
    if month in ['12','01','02']:
        return 4
    elif month in ['03','04','05']:
        return 1
    elif month in ['06','07','08']:
        return 2
    elif month in ['09','10','11']:
        return 3

train['season'] = train.month.apply(monthToSeason)
train.head()
#Making derived features
#train['date'] = train.datetime.apply(lambda x:x.split()[0])
train['hour'] = train.datetime.apply(lambda x:x.split()[1].split(':')[0])
#datetime.strptime(dateString,"%Y-%m-%d") => datetime.datetime(xxxx, x, x, 0, 0)
#datetime.strptime(dateString,"%Y-%m-%d").weekday() => 0~6 0:Mon, 6:Sun
#calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()] => return fullname of weekday
#e.g) calendar.day_name[datetime.strptime(train.date.min(),"%Y-%m-%d").weekday()] => 'Saturday'
train['weekday'] = train.date.apply(lambda dateString: calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
train['month'] = train.date.apply(lambda dateString: dateString.split("-")[1])
train['season'] = train.season.map({1:"Spring",2:"Summer",3:"Autumn",4:"Winter"})
train['weather'] = train.weather.map({1:"Clear, Few clouds, Partly cloudy, Partly cloudy",2:"Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
                                     3:"Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",4:"Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"})

train.info()
#Changing the datatype to 'Category' what's proper one to be converted
ChangingtoCat = ['season','holiday','workingday','weather','hour','weekday','month']
for var in ChangingtoCat:
    train[var] = train[var].astype('category')
train.info()
#Dropping the useless value in our dataset
train = train.drop('datetime',1)
#Looking at the count of dtypes in dataset
df_datatypes= pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={'index':'variableType',0:'count'})
fig = plt.figure(figsize=[12,5])
ax = fig.subplots()
sns.barplot(x='variableType',y='count',data=df_datatypes, ax=ax)
ax.set(xlabel='variableTypeariable Type', ylabel='Count',title="Variables DataType Count")
train.columns
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.boxplot(x='season',y='count',data=train,order=['Spring','Summer','Autumn','Winter'])
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.boxplot(x='month',y='count',data=train)
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.boxplot(x='hour',y='count',data=train)
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.boxplot(x='weekday',y='count',data=train,order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

tab = train.groupby('holiday',as_index=False)['count'].sum()
tab.index = tab['holiday']
tab = tab.drop('holiday',1)
tab.div(tab.sum(0),1).plot(kind='bar',stacked=True)
tab
tab = train.groupby('workingday',as_index=False)['count'].sum()
tab.index = tab['workingday']
tab = tab.drop('workingday',1)
tab.div(tab.sum(0),1).plot(kind='bar',stacked=True)
sns.boxplot(x='workingday',y='count',data=train)
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.distplot(train.temp,kde=False,bins=range(train.temp.min().astype('int'),train.temp.max().astype('int')+1,1))
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.distplot(train.atemp,kde=False,bins=range(train.atemp.min().astype('int'),train.atemp.max().astype('int')+1,1))
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.distplot(train.humidity,kde=False,bins=range(train.humidity.min(),train.humidity.max()+1,1))
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.distplot(train.windspeed,kde=False,bins=range(train.windspeed.min().astype('int'),train.windspeed.max().astype('int')+1,1))
#remove outlier
withoutOutlier = train[np.abs(train['count']-train['count'].mean())<=(3*train['count'].std())]
withoutOutlier.shape
train.shape
fig = plt.figure(figsize=[20,10])
cols = ['temp','atemp','humidity','windspeed','casual','registered','count']
sns.heatmap(train.loc[:,cols].corr(),annot=True,square=True,vmax=0.8)
"""Visualizing Distribution Of Data
As it is visible from the below figures that "count" variable is skewed towards right. 
It is desirable to have Normal distribution as most of the machine learning techniques require
dependent variable to be Normal. One possible solution is to take log transformation on "count" variable 
after removing outlier data points. After the transformation the data looks lot better 
but still not ideally following normal distribution."""

fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.distplot(train['count'])
ax2 = fig.add_subplot(2,2,2)
ax2 = stats.probplot(train['count'],dist='norm',fit=True)
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.distplot(np.log(withoutOutlier['count']))
ax4 = fig.add_subplot(2,2,4)
ax4 = stats.probplot(np.log1p(withoutOutlier['count']),dist='norm',fit=True)
train.groupby(['hour','season'])['count'].mean().reset_index()
fig = plt.figure(figsize=[12,20])
ax1 = fig.add_subplot(4,1,1)
df_month_grouped = train.groupby('month')['count'].mean().reset_index()
ax1 = sns.barplot(x='month',y='count',data=df_month_grouped)
ax2 = fig.add_subplot(4,1,2)
df_hour_season_grouped= train.groupby(['hour','season'])['count'].mean().reset_index()
ax2 = sns.pointplot(x='hour',y='count',hue='season',data=df_hour_season_grouped)
ax3 = fig.add_subplot(4,1,3)
df_hour_week_grouped = train.groupby(['hour','weekday'])['count'].mean().reset_index()
ax3 = sns.pointplot(x='hour',y='count',hue='weekday',data=df_hour_week_grouped)
ax4 = fig.add_subplot(4,1,4)
df_hour_species_melted = pd.melt(train[['hour','casual','registered']],id_vars='hour')
df_hour_species_grouped = df_hour_species_melted.groupby(['hour','variable'])['value'].mean().reset_index()
ax4 = sns.pointplot(x='hour',y='value',hue='variable',data=df_hour_species_grouped)
#Making the combined set for adjusted features
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#train_casual_registered_count = train[["casual","registered","count"]]
#train = train.drop(["casual","registered","count"],axis=1)
combine = pd.concat([train,test])
combine['date'] = combine.datetime.apply(lambda x: x.split()[0])
combine['hour'] = combine.datetime.apply(lambda x: x.split()[1].split(":")[0])
combine['weekday'] = combine.date.apply(lambda dateString: calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
combine['year'] = combine.date.apply(lambda x: x.split("-")[0])
combine['month'] = combine.date.apply(lambda x: x.split("-")[1])
combine['season'] = combine.month.apply(monthToSeason)
from sklearn.ensemble import RandomForestRegressor

dataWind0 = combine[combine['windspeed']==0]
dataWindNot0 = combine[combine['windspeed']!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ['season','weather','humidity','month','temp','year','atemp']
rfModel_wind.fit(dataWindNot0[windColumns],dataWindNot0['windspeed'])

wind0Values = rfModel_wind.predict(dataWind0[windColumns])
dataWind0['windspeed'] = wind0Values
combine = pd.concat([dataWindNot0,dataWind0])
combine.reset_index(inplace=True)
combine.drop('index',inplace=True,axis=1)
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]
for var in categoricalFeatureNames:
    combine[var] = combine[var].astype('category')
train = combine[pd.notnull(combine['count'])].sort_values(by=['datetime'])
test = combine[~pd.notnull(combine['count'])].sort_values(by=['datetime'])
datetimecol = test["datetime"]
yLabels = train["count"]
yLablesRegistered = train["registered"]
yLablesCasual = train["casual"]
train  = train.drop(dropFeatures,axis=1)
test  = test.drop(dropFeatures,axis=1)
train['weekday'].cat.categories = [0,1,2,3,4,5,6]
test['weekday'].cat.categories = [0,1,2,3,4,5,6]
"""
RMSLE
과대평가 된 항목보다는 과소평가 된 항목에 페널티를 주는방식
오차를 제곱하여 형균한 값의 제곱근으로 값이 작아질 수록 정밀도가 높음
0에 가까운 값이 나올 수록 정밀도가 높다
"""
# y is predict value y_ is actual value
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y), 
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
train.info()
#Linear Regression Model
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore',category=DeprecationWarning)

lr = LinearRegression()

yLabelslog = np.log1p(yLabels)
lr.fit(train,yLabelslog)

preds = lr.predict(train)
print('RMSLE Value For Linear Regression: {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
#Ridge
ridge = Ridge()
ridge_params = {'max_iter':[3000],'alpha':[0.1,1,2,3,4,10,30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle,greater_is_better=False)
grid_ridge = GridSearchCV(ridge,ridge_params,scoring=rmsle_scorer,cv=5)

grid_ridge.fit(train,yLabelslog)
preds = grid_ridge.predict(train)
print(grid_ridge.best_params_)
print('RMSLE Value for Ridge Regression '.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))

fig = plt.figure(figsize=[12,10])
df = pd.DataFrame(grid_ridge.cv_results_)
df['alpha'] = df.params.apply(lambda x:x['alpha'])
df['rmsle'] = df.mean_test_score.apply(lambda x:-x)
sns.pointplot(x='alpha',y='rmsle',data=df)
#randomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_params = {'n_estimators':[1,10,100]}
grid_rf = GridSearchCV(rf,rf_params,scoring=rmsle_scorer, cv=5)

grid_rf.fit(train,yLabelslog)
preds = grid_rf.predict(train)
print(grid_rf.best_params_)
print('RMSLE Value For Random Forest: {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01);
gbm.fit(train,yLabelslog)
preds = gbm.predict(train)
print('RMSLE Value For Gradient Boosting: {}'.format(rmsle(np.exp(yLabelslog),np.exp(preds),False)))
predsTest = gbm.predict(test)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(yLabels,ax=ax1,bins=50)
sns.distplot(np.exp(predsTest),ax=ax2,bins=50)
submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })
submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)