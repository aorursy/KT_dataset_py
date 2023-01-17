# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv',parse_dates = ['Date'])

test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

submission = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')
train.head()
train.info()
date_data = train['Date']

confirmed_cases = train['ConfirmedCases']

plt.figure(figsize=(10,8))

plt.plot(date_data,confirmed_cases)

plt.xticks(rotation=90)

plt.title('Time Series Analysis for the confirmed Cases')

plt.show()
date_data = train['Date']

Fatalities = train['Fatalities']

plt.figure(figsize=(10,8))

plt.plot(date_data,Fatalities)

plt.xticks(rotation=90)

plt.title('Time Series Analysis for Fatalities')

plt.show()
train.head()
train['Province/State'].value_counts()
plt.figure(figsize=(20,5))

sns.countplot(y = train['ConfirmedCases'])

plt.title('Count for confirmed cases')

plt.show()
plt.title('Count for confirmed cases')

sns.distplot(train['ConfirmedCases'],kde = False,bins=20)
train_new = train[train['ConfirmedCases'] > 0]

train_new
plt.figure(figsize=(10,8))

sns.barplot(x='Date',y='ConfirmedCases',data=train_new)

plt.xticks(rotation=45)

plt.title('Confirmed cases as per Date')

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(x='Date',y='Fatalities',data=train_new)

plt.xticks(rotation=45)

plt.title('Confirmed Death as per Date')

plt.show()
train_new.head()
train_new['Week'] = train_new['Date'].dt.week

train_new['Day'] = train_new['Date'].dt.day

train_new['DayOfWeek'] = train_new['Date'].dt.dayofweek

train_new['DayOfYear'] = train_new['Date'].dt.dayofyear

train_new.head()
df = train_new[['Date','Week','Day','DayOfWeek','DayOfYear','ConfirmedCases','Fatalities']]

df.head()
from sklearn.linear_model import LinearRegression,Lasso

from sklearn.linear_model import BayesianRidge

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,roc_auc_score

from sklearn.model_selection import train_test_split
X = df.drop(['Date','ConfirmedCases','Fatalities'],axis=1)

y = df[['ConfirmedCases','Fatalities']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(f'Size of X_train : {X_train.shape}')

print(f'Size of X_test : {X_test.shape}')

print(f'Size of y_train : {y_train.shape}')

print(f'Size of y_test : {y_test.shape}')
X_train.head()
y_train.head()
def predict_confirmed_cases(regression_algo):

    r = regression_algo()

    r.fit(X_train,y_train['ConfirmedCases'])

    y_pred = r.predict(X_test)

    rSquare = r2_score(y_test['ConfirmedCases'],y_pred)

    confirmed_cases.append(rSquare)



def predict_confirmed_deths(algos):

    r = algos()

    r.fit(X_train,y_train['Fatalities'])

    y_pred = r.predict(X_test)

    rSquare = r2_score(y_test['Fatalities'],y_pred)

    confirmed_death.append(rSquare)

    

models = [KNeighborsRegressor,LinearRegression,RandomForestRegressor,DecisionTreeRegressor,BayesianRidge,

          GradientBoostingRegressor,Lasso]



confirmed_cases = []

confirmed_death = []
for i in models:

    predict_confirmed_cases(i)
for j in models:

    predict_confirmed_deths(j)
confirmed_cases
confirmed_death
models = pd.DataFrame({

    'Model': ["KNeighborsRegressor","LinearRegression","RandomForestRegressor","DecisionTreeRegressor","BayesianRidge",

          "GradientBoostingRegressor","Lasso"],

    'ConfirmedCase_r2': confirmed_cases,

    'Fatalities_r2' : confirmed_death

})
models
test.head()
test.info()
test_data = test[['ForecastId','Date']]

test_data.head()
test_data['Date'] = pd.to_datetime(test_data['Date'])

test_data['Week'] = test_data['Date'].dt.week

test_data['Day'] = test_data['Date'].dt.day

test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek

test_data['DayOfYear'] = test_data['Date'].dt.dayofyear

test_data.head()
Kneighbour = KNeighborsRegressor()

Kneighbour.fit(X_train,y_train['ConfirmedCases'])
decisiontree = GradientBoostingRegressor()

decisiontree.fit(X_train,y_train['Fatalities'])
test_data['ConfirmedCases'] = Kneighbour.predict(test_data.drop(['Date','ForecastId'],axis=1))
test_data['Fatalities'] = decisiontree.predict(test_data.drop(['Date','ForecastId','ConfirmedCases'],axis=1))
test_data.head()
test_data = test_data[['ForecastId','ConfirmedCases','Fatalities']]
test_data.head()
test_data.to_csv('submission.csv',index=False)