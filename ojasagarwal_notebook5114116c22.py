import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
dtrain=pd.read_csv('../input/train.csv')

dtest=pd.read_csv('../input/test.csv')
dtrain.shape
dtrain.head(3)
dtrain.dtypes
d=dtrain.append(dtest)

d.reset_index(inplace=True)

d.drop('index',inplace=True,axis=1)

d["date"] = d.datetime.apply(lambda x : x.split()[0])

d["hour"] = d.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

d["year"] = d.datetime.apply(lambda x : x.split()[0].split("-")[0])

d["weekday"] = d.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

d["month"] = d.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

dwind0=d[d['windspeed']==0]

dwindnot0=d[d['windspeed']!=0]

wind_model=RandomForestRegressor()

field=['season','weather','humidity','month','temp','year']

wind_model.fit(dwindnot0[field],dwindnot0['windspeed'])

values_wind=wind_model.predict(X=dwind0[field])

dwind0['windspeed']=values_wind

data=dwindnot0.append(dwind0)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)
cat_var_list=["season","holiday","workingday","weather","weekday","month","year","hour"]

for i in cat_var_list:

    data[i]=data[i].astype('category')
dtrain=data[pd.notnull(data['count'])].sort_values(by=['datetime'])

dtest=data[~pd.notnull(data['count'])].sort_values(by=['datetime'])
%matplotlib inline

fig,axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sn.boxplot(data=dtrain,y="count",orient="v",ax=axes[0][0])

sn.boxplot(data=dtrain,y="count",x="season",orient="v",ax=axes[0][1])

sn.boxplot(data=dtrain,y="count",x="hour",orient="v",ax=axes[1][0])

sn.boxplot(data=dtrain,y="count",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Over Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Over Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Over Working Day")



%matplotlib inline

corrMatt = dtrain[["temp","atemp","humidity","windspeed","count"]].corr()

fig,ax= plt.subplots()

fig.set_size_inches(10,5)

sn.heatmap(corrMatt,vmax=1,square=True,annot=True)

%matplotlib inline

fig,axes = plt.subplots()

fig.set_size_inches(6, 5)

sn.distplot(dtrain["count"])

fig,axes = plt.subplots()

fig.set_size_inches(6,5)

sn.distplot(np.log(dtrain["count"]))

drop_list=['casual','registered','datetime','count','date']



y_val=dtrain['count']

y_log=np.log(y_val)

date_time=dtest['datetime']

dtrain=dtrain.drop(drop_list,axis=1)

dtest=dtest.drop(drop_list,axis=1)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(dtrain,y_log,test_size=0.2,random_state=7)
lin_model=LinearRegression()

lin_model.fit(X=Xtrain,y=Ytrain)

predict_linear=lin_model.predict(X=Xtest) 

print("RMSLE for Linear Regressor: ",rmsle(np.exp(Ytest),np.exp(predict_linear)))

rf_model=RandomForestRegressor(n_estimators=500,random_state=7)

rf_model.fit(X=Xtrain,y=Ytrain)

predict_rf=rf_model.predict(X=Xtest)

print("RMSLE for Random Forest Regressor: ",rmsle(np.exp(Ytest),np.exp(predict_rf)))
gb_model = GradientBoostingRegressor(n_estimators=500,random_state=7)

gb_model.fit(Xtrain,Ytrain)

predict_gb = gb_model.predict(X=Xtest)

print("RMSLE for Gradient Boosting Regressor: ",rmsle(np.exp(Ytest),np.exp(predict_gb)))
Xtrain,Xtest,Ytrain,Ytest=train_test_split(dtrain,y_val,test_size=0.2,random_state=7)
lin_model2=LinearRegression()

lin_model2.fit(Xtrain,Ytrain)

predict_linear2=lin_model2.predict(Xtest)

print("RMSLE for Linear Regressor without log: ",rmsle(Ytest,predict_linear2))
rf_model2=RandomForestRegressor(n_estimators=500,random_state=7)

rf_model2.fit(Xtrain,Ytrain)

predict_rf2=rf_model2.predict(X=Xtest)

print("RMSLE for Random Forest Regressor without log: ",rmsle(Ytest,predict_rf2))
# =============================================================================

# pred_test=gb_model.predict(X=dtest)

# output=pd.DataFrame({

#         'datetime': date_time,

#         'count': [max(0,x)for x in np.exp(pred_test)]

#         })

# print output.head(3)

# output.to_csv('gb.csv')

# 

# =============================================================================
