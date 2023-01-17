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
X_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

X_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')



X_train.rename(columns={'Country_Region':'Country'}, inplace=True)

X_test.rename(columns={'Country_Region':'Country'}, inplace=True)



X_train.rename(columns={'Province_State':'State'}, inplace=True)

X_test.rename(columns={'Province_State':'State'}, inplace=True)



X_train.Date = pd.to_datetime(X_train.Date, infer_datetime_format=True)



X_test.Date = pd.to_datetime(X_test.Date, infer_datetime_format=True)



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
X_xTrain = X_train.copy()



X_xTrain.State.fillna(EMPTY_VAL, inplace=True)

X_xTrain.State = X_xTrain.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTrain.loc[:, 'Date'] = X_xTrain.Date.dt.strftime("%m%d")

X_xTrain.Date  = X_xTrain.Date.astype(int)



X_xTest = X_test.copy()



X_xTest.State.fillna(EMPTY_VAL, inplace=True)

X_xTest.State = X_xTest.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTest.loc[:, 'Date'] = X_xTest.Date.dt.strftime("%m%d")

X_xTest.Date  = X_xTest.Date.astype(int)



#X_xTrain['Date'] = pd.to_datetime(X_xTrain['Date'],infer_datetime_format=True)

#X_xTest['Date'] = pd.to_datetime(X_xTest['Date'],infer_datetime_format=True)



#X_xTrain['Day_of_Week'] = X_xTrain['Date'].dt.dayofweek

#X_xTest['Day_of_Week'] = X_xTest['Date'].dt.dayofweek



#X_xTrain['Month'] = X_xTrain['Date'].dt.month

#X_xTest['Month'] = X_xTest['Date'].dt.month



#X_xTrain['Day'] = X_xTrain['Date'].dt.day

#X_xTest['Day'] = X_xTest['Date'].dt.day



#X_xTrain['Day_of_Year'] = X_xTrain['Date'].dt.dayofyear

#X_xTest['Day_of_Year'] = X_xTest['Date'].dt.dayofyear



#X_xTrain['Week_of_Year'] = X_xTrain['Date'].dt.weekofyear

#X_xTest['Week_of_Year'] = X_xTest['Date'].dt.weekofyear



#X_xTrain['Quarter'] = X_xTrain['Date'].dt.quarter  

#X_xTest['Quarter'] = X_xTest['Date'].dt.quarter  



#X_xTrain.drop('Date',1,inplace=True)

#X_xTest.drop('Date',1,inplace=True)
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



X_xTrain.Country = le.fit_transform(X_xTrain.Country)

X_xTrain.State = le.fit_transform(X_xTrain.State)



X_xTest.Country = le.fit_transform(X_xTest.Country)

X_xTest.State = le.fit_transform(X_xTest.State)
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn import preprocessing



le = preprocessing.LabelEncoder()



from xgboost import XGBRegressor

#from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor



countries = X_xTrain.Country.unique()
# Predict data and Create submission file from test data

xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_xTrain.loc[X_xTrain.Country == country, :].State.unique()

    for state in states:

        X_xTrain_CS = X_xTrain.loc[(X_xTrain.Country == country) & (X_xTrain.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_xTrain_CS = X_xTrain_CS.loc[:, 'ConfirmedCases']

        y2_xTrain_CS = X_xTrain_CS.loc[:, 'Fatalities']

        

        X_xTrain_CS = X_xTrain_CS.loc[:, ['State', 'Country', 'Date']]

        #X_xTrain_CS = X_xTrain_CS.loc[:, ['State', 'Country', 'Day_of_Week', 'Month', 'Day', 'Day_of_Year', 'Week_of_Year', 'Quarter']]

        

        X_xTrain_CS.Country = le.fit_transform(X_xTrain_CS.Country)

        X_xTrain_CS.State = le.fit_transform(X_xTrain_CS.State)

        

        X_xTest_CS = X_xTest.loc[(X_xTest.Country == country) & (X_xTest.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        #X_xTest_CS = X_xTest.loc[(X_xTest.Country == country) & (X_xTest.State == state), ['State', 'Country', 'Day_of_Week', 'Month', 'Day', 'Day_of_Year', 'Week_of_Year', 'Quarter', 'ForecastId']]

        

        X_xTest_CS_Id = X_xTest_CS.loc[:, 'ForecastId']

        X_xTest_CS = X_xTest_CS.loc[:, ['State', 'Country', 'Date']]

        #X_xTest_CS = X_xTest_CS.loc[:, ['State', 'Country', 'Day_of_Week', 'Month', 'Day', 'Day_of_Year', 'Week_of_Year', 'Quarter']]

        

        X_xTest_CS.Country = le.fit_transform(X_xTest_CS.Country)

        X_xTest_CS.State = le.fit_transform(X_xTest_CS.State)

        

        #xmodel1 = XGBRegressor(n_estimators=1000)

        #xmodel1 = DecisionTreeClassifier()

        xmodel1 = DecisionTreeRegressor()

        xmodel1.fit(X_xTrain_CS, y1_xTrain_CS)

        y1_xpred = xmodel1.predict(X_xTest_CS)

        

        #xmodel2 = XGBRegressor(n_estimators=1000)

        #xmodel2 = DecisionTreeClassifier()

        xmodel2 = DecisionTreeRegressor()

        xmodel2.fit(X_xTrain_CS, y2_xTrain_CS)

        y2_xpred = xmodel2.predict(X_xTest_CS)

        

        xdata = pd.DataFrame({'ForecastId': X_xTest_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})

        xout = pd.concat([xout, xdata], axis=0)
xout.ForecastId = xout.ForecastId.astype('int')

xout.tail()

xout.to_csv('submission.csv', index=False)