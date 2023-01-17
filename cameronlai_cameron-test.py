import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing



import lightgbm as lgb

import xgboost as xgb
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")
display(train.head(5))

display(test.head(5))
train.rename(columns={'Country_Region':'Country'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)





train.rename(columns={'Province_State':'State'}, inplace=True)

test.rename(columns={'Province_State':'State'}, inplace=True)



train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)



y1_Train = train.iloc[:, -2]

display(y1_Train.head())

y2_Train = train.iloc[:, -1]

display(y2_Train.head())
X_Train = train.copy()

X_Test = test.copy()
X_Train.State.fillna("None", inplace=True)

X_Test.State.fillna("None", inplace=True)
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



X_Train['Country'] = le.fit_transform(X_Train['Country'])

X_Train['State'] = le.fit_transform(X_Train['State'])

X_Train["Date"]  = X_Train["Date"].astype(int)



display(X_Train.head())



X_Test['Country'] = le.fit_transform(X_Test['Country'])

X_Test['State'] = le.fit_transform(X_Test['State'])

X_Test["Date"]  = X_Test["Date"].astype(int)



display(X_Test.tail())
countries = X_Train.Country.unique()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country, states)

    # check whether string is nan or not

    for state in states:

        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

        

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        

        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)

        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])

        

        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]

                

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]

        

        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)

        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])

        

        #models_C[country] = gridSearchCV(model, X_Train_CS, y1_Train_CS, param_grid, 10, 'neg_mean_squared_error')

        #models_F[country] = gridSearchCV(model, X_Train_CS, y2_Train_CS, param_grid, 10, 'neg_mean_squared_error')

        model1 = xgb.XGBRegressor(n_estimators=1000)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        

        model2 = xgb.XGBRegressor(n_estimators=1000)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        df_out = pd.concat([df_out, df], axis=0)

    # Done for state loop

# Done for country Loop

display(df_out.head())

display(df_out.shape)
df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()

df_out.to_csv(r'submission.csv', index=False)
df_out.head()