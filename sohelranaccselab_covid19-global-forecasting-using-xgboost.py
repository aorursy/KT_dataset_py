# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, KFold

from sklearn import ensemble

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
train.isnull().sum()
test.isnull().sum()

sample
sample['TargetValue'].sum()
train.sort_values(by=['TargetValue'])
last_date = train.Date.max()

df_countries = train[train['Date']==last_date]

df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()

df_countries = df_countries.nlargest(10,'TargetValue')

df_trend = train.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()

df_trend = df_trend.merge(df_countries, on='Country_Region')

df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)
train = train.drop(['County','Province_State','Target'],axis=1)

test = test.drop(['County','Province_State','Target'],axis=1)

train
last_date=train.Date.max()

df=train[train["Date"]==last_date]

df=df.groupby(by=["Country_Region"],as_index=False)["TargetValue"].sum()

countries=df.nlargest(5,"TargetValue")
cases=train.groupby(by=["Date","Country_Region"],as_index=False)["TargetValue"].sum()
cases=cases.merge(countries,on="Country_Region")

cases
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

train["Country_Region"]=le.fit_transform(train["Country_Region"])

test["Country_Region"]=le.fit_transform(test["Country_Region"])
train.Date=train.Date.apply(lambda x:x.split("-"))

test.Date=test.Date.apply(lambda x:x.split("-"))
def month_day(dataset):

    month=[]

    day=[]

    for i in dataset.Date:

        month.append(int(i[1]))

        day.append(int(i[2]))

    dataset["month"]=month

    dataset["day"]=day

    dataset=dataset.drop(["Date"],axis=1)

    return dataset
train=month_day(train)

test=month_day(test)

train.head()
y=train["TargetValue"].values
train = train.drop(['TargetValue', 'Id'], axis=1)

train
test = test.drop(['ForecastId'], axis=1)

test.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

xscale=scaler.fit_transform(train)

xscale



test=scaler.transform(test)
from xgboost import XGBRegressor

xgb=XGBRegressor()
from sklearn.model_selection import cross_val_score

#performance=cross_val_score(xgb,xscale,y,cv=10,scoring="neg_mean_absolute_error",n_jobs=-1)

#mae=-performance
xgb.fit(xscale,y)

prediction_xgb=xgb.predict(test)

prediction_xgb=np.around(prediction_xgb)

prediction_xgb
xgbpred=XGBRegressor(n_estimators=2000,max_depth=9, learning_rate=0.1, n_jobs=-1, reg_alpha=0.5, reg_lambda=1.5)

xgbpred.fit(xscale,y)

prediction=xgbpred.predict(test)
prediction=np.around(prediction)

prediction
test_copy=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

output = pd.DataFrame({'Id': test_copy.ForecastId  , 'TargetValue': prediction})

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