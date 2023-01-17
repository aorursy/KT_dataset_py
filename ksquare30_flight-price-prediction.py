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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
sns.set(style='darkgrid')

rcParams['figure.figsize']=10,6
pd.pandas.set_option('display.max_columns',None)
train=pd.read_excel(r'../input/flight-fare-prediction-mh/Data_Train.xlsx')

test=pd.read_excel(r'../input/flight-fare-prediction-mh/Test_set.xlsx')
train.head()
test.head()
train.isna().sum()
test.isna().sum()
train.shape
test.shape
train.info()
test.info()
train.dropna(inplace=True)
train.shape
chart=sns.countplot(x='Airline',data=train)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
chart=sns.barplot(x='Airline',y='Price',data=train)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
chart=sns.countplot(x='Airline',hue='Total_Stops',data=train)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
train['jou_day'] = pd.DatetimeIndex(train['Date_of_Journey']).day

train['jou_mon'] = pd.DatetimeIndex(train['Date_of_Journey']).month
chart=sns.countplot(x='Airline',hue='jou_mon',data=train)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
sns.barplot(x='jou_mon',y='Price',data=train)
test['jou_day'] = pd.DatetimeIndex(test['Date_of_Journey']).day

test['jou_mon'] = pd.DatetimeIndex(test['Date_of_Journey']).month
train['Airline'].value_counts()
test['Airline'].value_counts()
train['Destination'].unique()
test['Destination'].unique()
train_airline=pd.get_dummies(train.Airline,drop_first=True)
train_airline=train_airline.drop(['Trujet'],axis=1)
train_airline.head()
test_airline=pd.get_dummies(test.Airline,drop_first=True)
test_airline.head()
train_source=pd.get_dummies(train.Source,drop_first=True)

print(train_source.head())



train_destination=pd.get_dummies(train.Destination,drop_first=True)

print(train_destination.head())
test_source=pd.get_dummies(test.Source,drop_first=True)

print(test_source.head())



test_destination=pd.get_dummies(test.Destination,drop_first=True)

print(test_destination.head())
train['Total_Stops'].unique()
stops={'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}



train['Total_Stops']=train['Total_Stops'].map(stops)
train.head()
test['Total_Stops'].unique()
stops={'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}



test['Total_Stops']=test['Total_Stops'].map(stops)
train['dep_hour']=pd.to_datetime(train['Dep_Time']).dt.hour

train['dep_min']=pd.to_datetime(train['Dep_Time']).dt.minute
test['dep_hour']=pd.to_datetime(test['Dep_Time']).dt.hour

test['dep_min']=pd.to_datetime(test['Dep_Time']).dt.minute
train['arr_hour']=pd.to_datetime(train['Arrival_Time']).dt.hour

train['arr_min']=pd.to_datetime(train['Arrival_Time']).dt.minute
test['arr_hour']=pd.to_datetime(test['Arrival_Time']).dt.hour

test['arr_min']=pd.to_datetime(test['Arrival_Time']).dt.minute
train.head()
test.head()
train.drop(['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Additional_Info'],axis=1,inplace=True)
test.drop(['Airline','Date_of_Journey','Source','Destination','Route','Dep_Time','Arrival_Time','Duration','Additional_Info'],axis=1,inplace=True)
train.head()
test.head()
train=pd.concat([train,train_airline,train_source,train_destination],axis=1)
train.head()
test=pd.concat([test,test_airline,test_source,test_destination],axis=1)
test.head()
train.shape
test.shape
X=train.drop(['Price'],axis=1)

y=train['Price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=5)
knr=KNeighborsRegressor(n_neighbors=2)

knr.fit(X_train,y_train)

y_pred=knr.predict(X_test)
score=r2_score(y_test,y_pred)

score
regressor=RandomForestRegressor(n_estimators = 100, random_state = 15)

regressor.fit(X_train,y_train)

y_Pred=regressor.predict(X_test)
scores=r2_score(y_test,y_Pred)

scores
train_pred=regressor.predict(test)

final_df=pd.DataFrame({ 'Price': train_pred })

final_df.to_csv('final_dataset.csv',index=False)