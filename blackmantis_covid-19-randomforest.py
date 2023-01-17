# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

import matplotlib.dates as dates

from datetime import datetime

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn import preprocessing, metrics

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load the train and test files

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

train.head()
#get information about the train data

train.info()
train = train.drop(columns=['County','Province_State'])

test  = test.drop(columns=['County','Province_State'])
train
#select the Country_Region and Target columns

train.iloc[:,[1,5]]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le1= LabelEncoder()

train.iloc[:,1] = le.fit_transform(train.iloc[:,1])

train.iloc[:,5] = le1.fit_transform(train.iloc[:,5])

test.iloc[:,1] = le.fit_transform(test.iloc[:,1])

test.iloc[:,5] = le1.fit_transform(test.iloc[:,5])
train['Date'] = pd.to_datetime(train['Date'])

train['Dayofweek'] = train['Date'].dt.dayofweek

train['Day'] = train['Date'].dt.day

train['Month'] = train['Date'].dt.month

train_1 = train.drop(columns=['Date'])
test['Date'] = pd.to_datetime(test['Date'])

test['Dayofweek'] = test['Date'].dt.dayofweek

test['Day'] = test['Date'].dt.day

test['Month'] = test['Date'].dt.month

test_df1 = test.drop(columns=['Date'])
#Save the ForecastId column in the test in a variable that will be used later for submission

f_id = test_df1['ForecastId']

test = test_df1.drop(columns=['ForecastId'])

test
#Select the target variable and the features to use for prediction

y_train = train_1['TargetValue']

x_train = train_1.drop(columns=['TargetValue','Id'])

x_train
test_1 = test[['Country_Region', 'Population', 'Weight', 'Target','Dayofweek','Day','Month']]

test_1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test_1 = sc.transform(test_1)
#Split the train data 

from sklearn.model_selection import train_test_split 



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#set parameters for model

rf = RandomForestRegressor(n_jobs = -2, n_estimators = 100)

#fit model

rf.fit(x_train , y_train)
prediction = rf.predict(x_test)
#check predicted outcome

check = pd.DataFrame({'pred':prediction})

check
#Check the accuracy score

accuracy =rf.score(x_test,y_test)

accuracy
predict =rf.predict(x_test_1)
#check predicted outcome for each row

predict[1563]
sub = pd.DataFrame({'id':f_id,'pred':predict})

sub
a=sub.groupby(['id'])['pred'].quantile(q=0.05).reset_index()

b=sub.groupby(['id'])['pred'].quantile(q=0.5).reset_index()

c=sub.groupby(['id'])['pred'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()