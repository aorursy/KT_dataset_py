# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import seaborn as sns

#import matplotlib.pyplot as plt

#import nltk

#from sklearn.preprocessing  import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler

#from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression

#from sklearn.naive_bayes import MultinomalNB

#from sklearn.svm import SVC

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.metrics import classification_report,confusion

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

#province state remove na

train['Province_State'].fillna(" ", inplace = True)

test['Province_State'].fillna(" ",inplace = True)
#adding the province with country

train['Country_Region']=train['Country_Region']+' '+train['Province_State']

test['Country_Region']=test['Country_Region']+' '+test['Province_State']

train.drop(['Province_State'],axis = 1 , inplace = True)

test.drop(['Province_State'],axis = 1 , inplace = True)
# Converting Date for use(strin to int)

split_data_train = train["Date"].str.split("-").to_list()

split_data_test = test["Date"].str.split("-").to_list()

train_date = pd.DataFrame(split_data_train, columns=["Year","Month","Date"])

test_date = pd.DataFrame(split_data_test, columns=["Year","Month","Date"])

del train_date["Year"]

del test_date["Year"]

train_date['Month']=train_date['Month'].astype(int)

test_date['Month']=test_date['Month'].astype(int)

train_date['Date']=train_date['Date'].astype(int)

test_date['Date']=test_date['Date'].astype(int)

del train["Date"]

del test["Date"]

train = pd.concat([train,train_date],axis=1)

test = pd.concat([test,test_date],axis=1)

train_Id = train["Id"]

del train["Id"]

test_Id = test["ForecastId"]

del test["ForecastId"]
train_x_full = train[['Country_Region','Month','Date']].copy()

train_y_full = train[['ConfirmedCases','Fatalities']].copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_x_full['Country_Region']=le.fit_transform(train_x_full['Country_Region'])

test['Country_Region']=le.transform(test['Country_Region'])
from sklearn.model_selection import train_test_split

x_train ,x_valid, y_train , y_valid = train_test_split(train_x_full , train_y_full , train_size = 0.85 , test_size = 0.15 )
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
model = LinearRegression()

model.fit(x_train,y_train)

preds = model.predict(x_valid)

print(mean_absolute_error(preds,y_valid))
model = RandomForestRegressor(n_estimators = 1000)

model.fit(x_train,y_train)

preds = model.predict(x_valid)

print(mean_absolute_error(preds,y_valid))
model = XGBRegressor(n_estimators = 500, max_depth = 35)

CC = y_train.ConfirmedCases

fatalities = y_train.Fatalities

model.fit(x_train, CC)

predCC = model.predict(x_valid)

model.fit(x_train, fatalities)

predfa = model.predict(x_valid)

p = pd.DataFrame({'ConfirmedCases': predCC, 'Fatalities' : predfa})

print(mean_absolute_error(y_valid, p))
CC = train_y_full.ConfirmedCases

fatalities = train_y_full.Fatalities

model.fit(train_x_full, CC)

predCC = model.predict(test)

model.fit(train_x_full, fatalities)

predfa = model.predict(test)

output = pd.DataFrame({'ForecastId': test_Id,'ConfirmedCases': predCC, 'Fatalities' : predfa})

output = np.around(output , decimals = 0)

output.to_csv('submission.csv', index=False)