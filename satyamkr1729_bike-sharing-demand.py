import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
train=pd.read_csv("../input/bike-sharing-demand/train.csv")
test=pd.read_csv("../input/bike-sharing-demand/test.csv")
combine=[train,test]
train.shape
train.head(10)
train.info()
train.describe()
for data in combine:
    data["dt"]=data["datetime"].apply(lambda x: x.split())
    data["hour"]=data["dt"].apply(lambda x: x[1].split(':')[0])
    data["ymd"]=data["dt"].apply(lambda x: x[0].split("-"))
    data["year"]=data["ymd"].apply(lambda x: x[0])
    data["month"]=data["ymd"].apply(lambda x: x[1])
    data["day"]=data["ymd"].apply(lambda x: x[2])
train=train.drop(["datetime","dt","ymd"],axis=1)
test=test.drop(["datetime","dt","ymd"],axis=1)
combine=[train,test]
for data in combine:
    data["year"]=data["year"].astype(int)
    data["month"]=data["month"].astype(int) 
    data["day"]=data["day"].astype(int)
    data["hour"]=data["hour"].astype(int)    
sns.distplot(train["count"],bins=20)
f,ax=plt.subplots(ncols=3,nrows=3)
f.set_size_inches(20,15)
sns.boxplot(y='count',data=train,ax=ax[0][0])
sns.boxplot(x='day',y='count',data=train,ax=ax[0][1])
sns.boxplot(x='hour',y='count',data=train,ax=ax[0][2])
sns.boxplot(x='season',y='count',data=train,ax=ax[1][0])
sns.boxplot(x='weather',y='count',data=train,ax=ax[1][1])
sns.boxplot(x='workingday',y='count',data=train,ax=ax[1][2])
sns.boxplot(x='holiday',y='count',data=train,ax=ax[2][0])

train=train.loc[np.abs(train['count']-train['count'].mean())<3*train['count'].std(),:]
f,ax=plt.subplots(ncols=3,nrows=3)
f.set_size_inches(20,15)
sns.boxplot(y='count',data=train,ax=ax[0][0])
sns.boxplot(x='day',y='count',data=train,ax=ax[0][1])
sns.boxplot(x='hour',y='count',data=train,ax=ax[0][2])
sns.boxplot(x='season',y='count',data=train,ax=ax[1][0])
sns.boxplot(x='weather',y='count',data=train,ax=ax[1][1])
sns.boxplot(x='workingday',y='count',data=train,ax=ax[1][2])
sns.boxplot(x='holiday',y='count',data=train,ax=ax[2][0])

sns.distplot(train["count"],bins=20)
train.shape
train[["month","count"]].groupby(["month"],as_index=False).mean().sort_values(by='count',ascending=True)
train[["day","count"]].groupby(["day"],as_index=False).mean().sort_values(by='count',ascending=True)
train[["year","count"]].groupby(["year"],as_index=False).mean().sort_values(by='count',ascending=True)
train[["hour","count"]].groupby(["hour"],as_index=True).mean().sort_values(by='hour',ascending=True)
train[["workingday","count"]].groupby(["workingday"],as_index=False).mean().sort_values(by='count',ascending=True)
train[["holiday","count"]].groupby(["holiday"],as_index=False).mean().sort_values(by='count',ascending=True)
train[["season","count"]].groupby(["season"],as_index=False).mean().sort_values(by='count',ascending=True)
train[['weather','count']].groupby(['weather'],as_index=False).mean().sort_values(by='count',ascending=True)
train.columns
c_features=['temp','atemp','humidity','windspeed','casual','registered','count','hour']
f,ax= plt.subplots(figsize=(18,18))
sns.heatmap(train[c_features].corr(),annot=True,fmt='.1g',linewidth=.2,ax=ax)
sns.violinplot(x="season",y="count",hue="year",data=train,split=True)
sns.violinplot(x="weather",y="count",data=train,hue="year",split=True)
train['weather'].value_counts()
sns.violinplot(x="workingday",y="count",data=train,hue="year")
sns.violinplot(x='holiday',y='count',data=train,hue='year')
sns.barplot(x='year',y='count',data=train)
g=sns.FacetGrid(train,col='season',aspect=1.5)
g.map(sns.violinplot,'weather','count','year',split=True)
f,ax=plt.subplots(nrows=5)
f.set_size_inches(20,27)
sns.pointplot(x="day",y="count",data=train,hue='year',ax=ax[0])
sns.pointplot(x="day",y="count",data=train,hue='season',ax=ax[1])
sns.pointplot(x="day",y="count",data=train,hue='weather',ax=ax[2])
sns.pointplot(x="day",y="count",data=train,hue='workingday',ax=ax[3])
sns.pointplot(x="day",y="count",data=train,hue='holiday',ax=ax[4])
f,ax=plt.subplots(nrows=5)
f.set_size_inches(20,27)
sns.pointplot(x="hour",y="count",data=train,hue='year',ax=ax[0])
sns.pointplot(x="hour",y="count",data=train,hue='season',ax=ax[1])
sns.pointplot(x="hour",y="count",data=train,hue='weather',ax=ax[2])
sns.pointplot(x="hour",y="count",data=train,hue='workingday',ax=ax[3])
sns.pointplot(x="hour",y="count",data=train,hue='holiday',ax=ax[4])
sns.distplot(train['humidity'],bins=20)
sns.distplot(train['atemp'],bins=20)
f,ax=plt.subplots(ncols=2)
f.set_size_inches(12,10)
sns.regplot(x="atemp",y="count",data=train,ax=ax[0])
sns.regplot(x="humidity",y="count",data=train,ax=ax[1])
train.columns
test.columns
train=train.drop(['temp','casual','registered'],axis=1)
test=test.drop(['temp'],axis=1)
combine=[train,test]
train_ws=train[train['windspeed']!=0]
train_ws_fill=train.loc[train['windspeed']==0,:]
rfr=RandomForestRegressor(n_estimators=20,random_state=10)
rfr.fit(train_ws.drop(['windspeed'],axis=1),train_ws['windspeed'])
train_ws_fill=train_ws_fill.copy()
train_ws_fill.loc[:,'windspeed']=rfr.predict(train_ws_fill.drop(['windspeed'],axis=1))
train_ws_fill.head()
train.loc[train['windspeed']==0,'windspeed']=train_ws_fill['windspeed']
train.head(10)
sns.regplot(x='windspeed',y='count',data=train)
train['count']=train['count'].apply(lambda x: np.sqrt(x))
sns.distplot(train['count'],bins=20)
train['year']=train['year'].apply(lambda x: 1 if x==2012 else 0).astype(int)
train['hour_band']=pd.cut(x=train['hour'],bins=5,precision=0)
train['hour_band'].unique()
train=train.copy()
train.loc[train['hour']<=5.0,'hour']=1
train.loc[(train['hour']>5.0) & (train['hour']<=9.0),'hour']=2
train.loc[(train['hour']>9.0) & (train['hour']<=14.0),'hour']=3
train.loc[(train['hour']>14.0) & (train['hour']<=18.0),'hour']=4
train.loc[(train['hour']>18.0),'hour']=5
train=train.drop(['hour_band'],axis=1)
train.head()
train,test=train_test_split(train,test_size=0.20)
train.shape
test.shape
x_train=train.drop(['count'],axis=1)
y_train=train['count']
lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(test.drop(['count'],axis=1))
mse(test['count'],lr_pred)

