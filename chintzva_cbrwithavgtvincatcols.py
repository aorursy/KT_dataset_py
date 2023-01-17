# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
data=covid_data
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

data['Date']=data['Date']-data['Date'][0]

data['Date']=data['Date'].dt.days
lb=LabelEncoder()

data['Target']=lb.fit_transform(data['Target'])
data.drop(['Id'],axis=1,inplace=True)
y=data.TargetValue
a,b,y_train,y_val=train_test_split(data,y,test_size=0.2,random_state=0)
X_train=a.copy()

X_val=b.copy()
counties=X_train['County'].unique()
for county in counties:

    X_train['County'][(X_train['County']==county) & (X_train['Target']==0)]=X_train['TargetValue'][(X_train['County']==county) & (X_train['Target']==0)].mean()

    X_train['County'][(X_train['County']==county) & (X_train['Target']==1)]=X_train['TargetValue'][(X_train['County']==county) & (X_train['Target']==1)].mean()
states=X_train['Province_State'].unique()
for state in states:

    X_train['Province_State'][(X_train['Province_State']==state) & (X_train['Target']==0)]=X_train['TargetValue'][(X_train['Province_State']==state) & (X_train['Target']==0)].mean()

    X_train['Province_State'][(X_train['Province_State']==state) & (X_train['Target']==1)]=X_train['TargetValue'][(X_train['Province_State']==state) & (X_train['Target']==1)].mean()
Countries=data['Country_Region'].unique()
for country in Countries:

    X_train['Country_Region'][(X_train['Country_Region']==country) & (X_train['Target']==0)]=X_train['TargetValue'][(X_train['Country_Region']==country) & (X_train['Target']==0)].mean()

    X_train['Country_Region'][(X_train['Country_Region']==country) & (X_train['Target']==1)]=X_train['TargetValue'][(X_train['Country_Region']==country) & (X_train['Target']==1)].mean()
X_train=X_train.bfill(axis=1)
X_train.drop(['TargetValue'],axis=1,inplace=True)
X_val.drop(['TargetValue'],axis=1,inplace=True)
for county in counties:

    X_val['County'][(X_val['County']==county) & (X_val['Target']==0)]=a['TargetValue'][(a['County']==county) & (a['Target']==0)].mean()

    X_val['County'][(X_val['County']==county) & (X_val['Target']==1)]=a['TargetValue'][(a['County']==county) & (a['Target']==1)].mean()
for state in states:

    X_val['Province_State'][(X_val['Province_State']==state) & (X_val['Target']==0)]=a['TargetValue'][(a['Province_State']==state) & (a['Target']==0)].mean()

    X_val['Province_State'][(X_val['Province_State']==state) & (X_val['Target']==1)]=a['TargetValue'][(a['Province_State']==state) & (a['Target']==1)].mean()
for country in Countries:

    X_val['Country_Region'][(X_val['Country_Region']==country) & (X_val['Target']==0)]=a['TargetValue'][(a['Country_Region']==country) & (a['Target']==0)].mean()

    X_val['Country_Region'][(X_val['Country_Region']==country) & (X_val['Target']==1)]=a['TargetValue'][(a['Country_Region']==country) & (a['Target']==1)].mean()
X_val=X_val.bfill(axis=1)
from catboost import CatBoostRegressor

model1=CatBoostRegressor()

model1.fit(X_train,y_train)
e1=y_val-model1.predict(X_val)

k1=y_train-model1.predict(X_train)

print('validation_pinball_loss_estimate',((e1[e1>=0]*0.05).sum()+(e1[e1<0]*(0.05-1)).sum())/len(y_val))

print('training_pinball_loss_estimate',((k1[k1>=0]*0.05).sum()+(k1[k1<0]*(0.05-1)).sum())/len(y_train))
print('validation_pinball_loss_estimate',((e1[e1>=0]*0.5).sum()+(e1[e1<0]*(0.5-1)).sum())/len(y_val))

print('training_pinball_loss_estimate',((k1[k1>=0]*0.5).sum()+(k1[k1<0]*(0.5-1)).sum())/len(y_train))
print('validation_pinball_loss_estimate',((e1[e1>=0]*0.95).sum()+(e1[e1<0]*(0.95-1)).sum())/len(y_val))

print('training_pinball_loss_estimate',((k1[k1>=0]*0.95).sum()+(k1[k1<0]*(0.95-1)).sum())/len(y_train))
covid_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
test=covid_test.copy()
test.drop(['ForecastId'],axis=1,inplace=True)
test['Target']=lb.transform(test['Target'])
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

initial_date=pd.to_datetime('2020-01-23',format='%Y-%m-%d')

test['Date']=test['Date']-initial_date

test['Date']=test['Date'].dt.days
for county in counties:

    test['County'][(test['County']==county) & (test['Target']==0)]=a['TargetValue'][(a['County']==county) & (a['Target']==0)].mean()

    test['County'][(test['County']==county) & (test['Target']==1)]=a['TargetValue'][(a['County']==county) & (a['Target']==1)].mean()
for state in states:

    test['Province_State'][(test['Province_State']==state) & (test['Target']==0)]=a['TargetValue'][(a['Province_State']==state) & (a['Target']==0)].mean()

    test['Province_State'][(test['Province_State']==state) & (test['Target']==1)]=a['TargetValue'][(a['Province_State']==state) & (a['Target']==1)].mean()
for country in Countries:

    test['Country_Region'][(test['Country_Region']==country) & (test['Target']==0)]=a['TargetValue'][(a['Country_Region']==country) & (a['Target']==0)].mean()

    test['Country_Region'][(test['Country_Region']==country) & (test['Target']==1)]=a['TargetValue'][(a['Country_Region']==country) & (a['Target']==1)].mean()
test=test.bfill(axis=1)
ans=model1.predict(test)
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')
sub1=sub.copy()
for i in range(len(ans)):

    sub1.iat[3*i,1]=ans[i]

    sub1.iat[3*i+1,1]=ans[i]

    sub1.iat[3*i+2,1]=ans[i]
sub1.to_csv('submission.csv',index=False)