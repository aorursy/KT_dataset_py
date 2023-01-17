# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
train.info()
train
train['date']=train.datetime.apply(lambda x: x.split()[0])
train['hour']=train.datetime.apply(lambda x:x.split()[1].split(':')[0])
train['weekday'] = train.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
train['month']=train.date.apply(lambda dateString: calendar.month_name[datetime.strptime(dateString,'%Y-%m-%d').month])
train['season']=train.season.map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})
train['weather']=train.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
category_vars=['hour','weekday','month','season','weather','holiday','workingday']
for var in category_vars:
    train[var]=train[var].astype('category')
train.info()
train=train.drop('datetime',axis=1)
df=pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={'index':'var_type',0:'count'})
fig,ax=plt.subplots()
sn.barplot(data=df,x='var_type',y='count',ax=ax)
ax.set(xlabel='var_type',ylabel='count')
train.isnull().sum()
fig,axes=plt.subplots(nrows=2,ncols=2)
sn.boxplot(data=train,y='count',orient='v',ax=axes[0][0])
sn.boxplot(data=train,y='count',x='season',orient='v',ax=axes[0][1])
sn.boxplot(data=train,y='count',x='hour',orient='v',ax=axes[1][0])
sn.boxplot(data=train,y='count',x='workingday',orient='v',ax=axes[1][1])
trainwo=train[np.abs(train['count']-train['count'].mean())<=3*train['count'].std()]
print('Shape of the DataFrame with outliers: ', train.shape)
print('Shape of the DataFrame without outliers: ', trainwo.shape)
corr=train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask=np.array(corr)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots()
sn.heatmap(corr,mask=mask,vmax=.8,square=True,annot=True)
fig,(ax1,ax2,ax3)=plt.subplots(ncols=3)
sn.regplot(x='temp',y='count',data=train,ax=ax1)
sn.regplot(x='windspeed',y='count',data=train,ax=ax2)
sn.regplot(x='humidity',y='count',data=train,ax=ax3)
fig,axes=plt.subplots(ncols=2,nrows=2)
sn.distplot(train['count'],ax=axes[0][0])
stats.probplot(train['count'],dist='norm',fit=True,plot=axes[0][1])
sn.distplot(np.log(trainwo['count']),ax=axes[1][0])
stats.probplot(np.log1p(trainwo['count']),dist='norm',fit=True,plot=axes[1][1])
fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4)
fig.set_size_inches(12,20)
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
month_averages=pd.DataFrame(train.groupby('month')['count'].mean()).reset_index()
month_averages_sorted=month_averages.sort_values(by='count',ascending=False)
sn.barplot(data=month_averages,x='month',y='count',ax=ax1,order=sortOrder)
hour_averages=pd.DataFrame(train.groupby(['hour','season'],sort=True)['count'].mean()).reset_index()
sn.pointplot(data=hour_averages,x=hour_averages['hour'],y=hour_averages['count'],hue=hour_averages['season'],join=True,ax=ax2)
hour_averages=pd.DataFrame(train.groupby(['hour','weekday'])['count'].mean()).reset_index()
sn.pointplot(data=hour_averages,x='hour',y='count',hue='weekday',hue_order=hueOrder,join=True,ax=ax3)
hourTransformed = pd.melt(train[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()
sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
hourTransformed
train=pd.read_csv('../input/train.csv')
test=pd.read_csv("../input/test.csv")
data=train.append(test)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

data['date']=data['datetime'].apply(lambda x:x.split()[0])
data['hour']=data['datetime'].apply(lambda x:x.split()[1].split(':')[0])
data['year']=data['date'].apply(lambda x:x.split('-')[0])
data['month']=data['date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d").month)
data['weekday']=data['date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d").weekday())
from sklearn.ensemble import RandomForestRegressor
wind0=data[data['windspeed']==0]
windNot0=data[data['windspeed']!=0]
rf_wind=RandomForestRegressor()
wind_cols=['season','weather','year','month','temp','atemp','humidity']
rf_wind.fit(windNot0[wind_cols],windNot0['windspeed'])
pred=rf_wind.predict(X=wind0[wind_cols])
wind0['windspeed']=pred
data=windNot0.append(wind0)
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)

categorical_features=['season','month','year','workingday','holiday','weather','hour']
numerical_features=['humidity','windspeed','temp','atemp']
drop_features=['casual','registered','datetime','date','count']
for var in categorical_features:
    data[var]=data[var].astype('category')
train=data[pd.notnull(data['count'])].sort_values(by=['datetime'])
test=data[~pd.notnull(data['count'])].sort_values(by='datetime')
datetimecol=test['datetime']
y_train=train['count']
y_train_registered=train['registered']
y_train_casual=train['casual']

train=train.drop(drop_features,axis=1)
test=test.drop(drop_features,axis=1)
def rmsle(y,y_,convertExp=True):
    if convertExp:
        y=np.exp(y)
        y_=np.exp(y_)
    log1=np.nan_to_num(np.array([np.log(p+1) for p in y]))
    log2=np.nan_to_num(np.array([np.log(a+1) for a in y_]))
    calc=(log1-log2)**2
    return np.sqrt(np.mean(calc))
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

lr=LinearRegression()
y_train_log=np.log1p(y_train)
lr.fit(train,y_train_log)
pred=lr.predict(train)
print("RMSLE Value For Linear Regression: ",rmsle(y_train_log,pred,True))

ridge=Ridge()
ridge_params_={'max_iter':[3000],'alpha':[0.1,1,2,3,4,10,30,100,200,300,400,800,900,1000]}
rmsle_scorer=metrics.make_scorer(rmsle,greater_is_better=False)
grid_ridge=GridSearchCV(ridge,ridge_params_,scoring=rmsle_scorer,cv=5)
y_train_log=np.log1p(y_train)
grid_ridge.fit(train,y_train_log)
preds=grid_ridge.predict(train)
print(grid_ridge.best_params_)
print("RMSLE Value For Ridge Regression: ", rmsle(y_train_log,preds,True))

fig,ax=plt.subplots()
df=pd.DataFrame(grid_ridge.cv_results_)
df
df['alpha']=df['param_alpha']
df['rmsle']=df['mean_test_score'].apply(lambda x:-x)
sn.pointplot(data=df,x='alpha',y='rmsle',ax=ax)
lasso=Lasso()
alpha=1/np.array([0.1,1,2,3,4,10,30,100,200,300,400,800,900,1000])
lasso_params={'max_iter':[3000],'alpha':alpha}

grid_lasso=GridSearchCV(lasso,lasso_params,scoring=rmsle_scorer,cv=5)
y_train_log=np.log1p(y_train)
grid_lasso.fit(train,y_train_log)
pred=grid_lasso.predict(train)
print(grid_lasso.best_params_)
print("RMSLE Value For Lasso Regression: ", rmsle(y_train_log,pred,True))

fig,ax=plt.subplots()
df=pd.DataFrame(grid_lasso.cv_results_)
df['alpha']=df['param_alpha']
df['rmsle']=-df['mean_test_score']
sn.pointplot(data=df,x='alpha',y='rmsle',ax=ax)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100)
y_train_log=np.log1p(y_train)
rf.fit(train,y_train_log)
pred=rf.predict(train)
print("RMSLE Value For Random Forest: ", rmsle(y_train_log,pred,True))
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
y_train_log=np.log1p(y_train)
gbr.fit(train,y_train_log)
pred=gbr.predict(train)
print("RMSLE Value For Gradient Boost: ", rmsle(y_train_log,pred,True))
pred_test=gbr.predict(test)
fig,(ax1,ax2)=plt.subplots(ncols=2)
sn.distplot(y_train,ax=ax1,bins=50)
sn.distplot(np.exp(pred_test),ax=ax2,bins=50)
submission=pd.DataFrame({
    'datetime':datetimecol,
    'count':[max(0,x) for x in np.exp(pred_test)]
})
submission.to_csv('bike_predictions_gbm.csv',index=False)
