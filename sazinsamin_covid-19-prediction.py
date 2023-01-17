import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df.head(3)
dt=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

dt.head(3)
ds=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

ds.head(3)
print('The train dataset shape : ',df.shape)

print('The test dataset shape : ',dt.shape)

print('The sunmission dataset shape : ',ds.shape)
df1=df.copy()

dt1=dt.copy()

ds1=ds.copy()
df1['Date']=pd.to_datetime(df['Date'])

dt1['Date']=pd.to_datetime(dt['Date'])
df2=df1.rename({'Country/Region':'Country','Province/State':'State','Id':'ForecastId'},axis=1)
dt2=dt1.rename({'Country/Region':'Country','Province/State':'State'},axis=1)
df2.head(2)
dt2.head(2)
print('The total missing values : ',df2.isnull().sum().sum())

print('The total missing values : ',dt2.isnull().sum().sum())
df2['State'].unique()
df2.fillna(0,inplace=True)
cases=df2.groupby('Date')['Date','ConfirmedCases','Fatalities'].sum().reset_index()
cases
fig1=px.line(cases,x='Date',y='ConfirmedCases',title='Cases Confirmed')
fig1
fig2=px.line(cases,x='Date',y="Fatalities",title='Death cases')

fig2
df2.drop(columns=['State'],inplace=True)
dt2.drop(columns=['State'],inplace=True)
df2.head(3)
dt2.head(3)
X_train=df2.drop(["Fatalities", "ConfirmedCases"],axis=1)

test=dt2.copy()
X_train=X_train.set_index(['Date'])

test=test.set_index(['Date'])
X_train.head(3)
def datetime_x(df):

    df['date'] = df.index

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear
datetime_x(X_train)
X_train.head(3)
X_train.drop(columns=['date'],inplace=True)
X_train.head(3)
datetime_x(test)

test.drop(columns=['date'],axis=1)
X_train.set_index(['ForecastId'])
test.set_index(['ForecastId'])
test.dtypes
X_train1=X_train.copy()
train_dummies=pd.get_dummies(X_train1['Country'],prefix='cr_')
train_dummies.head(3)
X_train2=pd.concat([X_train1,train_dummies],axis=1)
X_train2.head(3)
test_dummies=pd.get_dummies(test['Country'],prefix='cr_')
test2=pd.concat([test,test_dummies],axis=1)
test2.drop(columns=['date'],inplace=True)
test2.head(3)
X_train3=X_train2.drop(columns=['Country'],axis=1)

X_train3.head(3)
test3=test2.drop(columns=['Country'],axis=1)
target1=df['ConfirmedCases']

target2=df['Fatalities']
print('target 1 amount : ',target1.sum())

print('target 2 amount : ',target2.sum())
x_train,x_test,y_train,y_test=train_test_split(X_train3,target1,test_size=0.33,random_state=42)
from sklearn.tree import DecisionTreeRegressor  
rs= DecisionTreeRegressor(random_state = 0) 

rs.fit(x_train,y_train)

print('The score for',rs.score(x_test,y_test)*100)
rs.fit(X_train3,target1)
Confirmed=rs.predict(test3)
rs.fit(X_train3,target2)
Fatalities=rs.predict(test3)
sub=pd.DataFrame()

sub['ForecastId']=test3['ForecastId']

sub['ConfirmedCases']=Confirmed

sub['Fatalities']=Fatalities
sub.to_csv('submission.csv',index=False)