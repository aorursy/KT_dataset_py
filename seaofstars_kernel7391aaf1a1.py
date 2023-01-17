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



test = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv(r"/kaggle/input/covid19-global-forecasting-week-3/train.csv")
train.rename(columns={'Country_Region':'Country'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)



train.rename(columns={'Province_State':'State'}, inplace=True)

test.rename(columns={'Province_State':'State'}, inplace=True)



train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)



train.info()

test.info()



y1_xTrain = train.iloc[:, -2]

y1_xTrain.head()

y2_xTrain = train.iloc[:, -1]

y2_xTrain.head()
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
X_xTrain = train.copy()



X_xTrain['State'].fillna(EMPTY_VAL, inplace=True)

X_xTrain['State'] = X_xTrain.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTrain.loc[:, 'Date'] = X_xTrain.Date.dt.strftime("%m%d")

X_xTrain["Date"]  = X_xTrain["Date"].astype(int)



X_xTrain.tail()
X_xTest = test.copy()



X_xTest['State'].fillna(EMPTY_VAL, inplace=True)

X_xTest['State'] = X_xTest.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_xTest.loc[:, 'Date'] = X_xTest.Date.dt.strftime("%m%d")

X_xTest["Date"]  = X_xTest["Date"].astype(int)



X_xTest.tail()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



X_xTrain.Country = le.fit_transform(X_xTrain.Country)

X_xTrain['State'] = le.fit_transform(X_xTrain['State'])





X_xTest.Country = le.fit_transform(X_xTest.Country)

X_xTest['State'] = le.fit_transform(X_xTest['State'])

train.head()

train.loc[train.Country == 'Afghanistan', :]

test.tail()
from warnings import filterwarnings

filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



from xgboost import XGBRegressor



countries = X_xTrain.Country.unique()
# Predict data and Create submission file from test data

from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBRegressor

xout = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_xTrain.loc[X_xTrain.Country == country, :].State.unique()

    #print(country, states)

    # check whether string is nan or not

    for state in states:

        X_xTrain_CS = X_xTrain.loc[(X_xTrain.Country == country) & (X_xTrain.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_xTrain_CS = X_xTrain_CS.loc[:, 'ConfirmedCases']

        y2_xTrain_CS = X_xTrain_CS.loc[:, 'Fatalities']

        

        X_xTrain_CS = X_xTrain_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_xTrain_CS.Country = le.fit_transform(X_xTrain_CS.Country)

        X_xTrain_CS['State'] = le.fit_transform(X_xTrain_CS['State'])

        

        X_xTest_CS = X_xTest.loc[(X_xTest.Country == country) & (X_xTest.State == state), ['State', 'Country', 'Date', 'ForecastId']]

        

        X_xTest_CS_Id = X_xTest_CS.loc[:, 'ForecastId']

        X_xTest_CS = X_xTest_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_xTest_CS.Country = le.fit_transform(X_xTest_CS.Country)

        X_xTest_CS['State'] = le.fit_transform(X_xTest_CS['State'])

        

        param  = {'max_depth':range(2,7), 'min_child_weight':range(1,7),'reg_alpha':[0.001, 0.005, 0.01, 0.05, 1, 100],}

        

        #from sklearn.preprocessing import StandardScaler

        #X_xTrain = StandardScaler().fit_transform(X_xTrain)

        #X_xTest  = StandardScaler().fit_transform(X_xTest)

        

        x1 =XGBRegressor(n_estimators=1000)     

        x1.fit(X_xTrain_CS, y1_xTrain_CS)

        y1_xpred = x1.predict(X_xTest_CS)

        

        x2 = XGBRegressor(n_estimators=1000)    

        x2.fit(X_xTrain_CS, y2_xTrain_CS)

        y2_xpred = x2.predict(X_xTest_CS)

        

        print(X_xTrain_CS.shape)

        

        xdata = pd.DataFrame({'ForecastId': X_xTest_CS_Id, 'ConfirmedCases': y1_xpred, 'Fatalities': y2_xpred})

        xout = pd.concat([xout, xdata], axis=0)
xout.ForecastId = xout.ForecastId.astype('int')



xout.to_csv('submission.csv', index=False)
xout.tail()