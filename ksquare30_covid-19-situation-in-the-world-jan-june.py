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
import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
pd.pandas.set_option('display.max_columns',None)
train=pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
train.head()
test.head()
train.shape
test.shape
train.isna().sum()
test.isna().sum()
train.info()
test.info()
train.groupby('Country_Region')['Population'].max()
train['date']=pd.to_datetime(train.Date,format='%Y-%m-%d').dt.day

train['month']=pd.to_datetime(train.Date,format='%Y-%m-%d').dt.month
test['date']=pd.to_datetime(test.Date,format='%Y-%m-%d').dt.day

test['month']=pd.to_datetime(test.Date,format='%Y-%m-%d').dt.month
x=train.groupby('Country_Region')['Population'].max()

y=train.groupby('Country_Region')['Country_Region'].max()

fig=px.pie(data_frame=train,values=x,names=y,hole=0.30)

fig.update_traces(textposition='inside',textinfo='percent+label')
fig=px.pie(data_frame=train,values='TargetValue',names='Country_Region',hole=0.30)

fig.update_traces(textposition='inside',textinfo='percent+label')
plt.figure(figsize=(10,6))

sns.barplot(x='Target',y='TargetValue',data=train)
conf_train=train[train['Target']=='ConfirmedCases']

large10=conf_train.groupby('Country_Region')['TargetValue'].sum()

large10=large10.nlargest(10)

Large10=large10.to_frame()

Large10.reset_index(level=0, inplace=True)
plt.figure(figsize=(15,10))

sns.barplot(data=Large10,x='Country_Region',y='TargetValue')
conf_train=train[train['Target']=='Fatalities']

large10=conf_train.groupby('Country_Region')['TargetValue'].sum()

large10=large10.nlargest(10)

Large10=large10.to_frame()

Large10.reset_index(level=0, inplace=True)
plt.figure(figsize=(15,10))

sns.barplot(data=Large10,x='Country_Region',y='TargetValue')
top5=['US','United Kingdom','Brazil','Russia','India']

conf_train=train[train['Target']=='ConfirmedCases']

fdf=conf_train.loc[conf_train['Country_Region'].isin(top5)]

px.line(data_frame=fdf,y='TargetValue',x="Date",color='Country_Region')
conf_train=train[(train.Target=='ConfirmedCases') & (train.Country_Region=='India') ]

px.line(data_frame=conf_train,y='TargetValue',x="Date",color='Country_Region',hover_name="Target")
conf_train=train[(train.Target=='Fatalities') & (train.Country_Region=='India') ]

px.line(data_frame=conf_train,y='TargetValue',x="Date",color='Country_Region',hover_name="Target")
le=LabelEncoder()
train["Country_Region"]=le.fit_transform(train["Country_Region"])
test["Country_Region"]=le.fit_transform(test["Country_Region"])
train_id=train['Id']

train.drop(['County','Province_State','Target','Date','Id'],axis=1,inplace=True)
train.head()
test_id=test['ForecastId']

test.drop(['County','Province_State','Target','Date','ForecastId'],axis=1,inplace=True)
test.head()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=train.drop(['TargetValue'],axis=1)

y=train['TargetValue']
train_scaled=scaler.fit_transform(X)
test_scaled=scaler.transform(test)
xgbr=XGBRegressor(n_estimators=1500,max_depth=5)

xgbr.fit(train_scaled,y)
prediction=xgbr.predict(test_scaled)
output = pd.DataFrame({'Id': test_id  , 'TargetValue': prediction})

output.head()
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()