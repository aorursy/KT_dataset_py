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
PATH_WEEK3='/kaggle/input/covid19-global-forecasting-week-3'

df_train = pd.read_csv(f'{PATH_WEEK3}/train.csv')

df_test = pd.read_csv(f'{PATH_WEEK3}/test.csv')
#make it compact

df_train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)



df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train[df_train['Country']=='Brazil'].tail()
df_train[df_train['Country']=='Brazil'].plot(x='Date', y='ConfirmedCases')
import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.backends.backend_pdf import PdfPages



data1 = df_train[df_train['Country']=='Belgium']

plt.scatter(data1['Date'], np.log(data1['ConfirmedCases']+1))

data1 = df_train[df_train['Country']=='Japan']

plt.scatter(data1['Date'], np.log(data1['ConfirmedCases']+1))

data1 = df_train[df_train['Country']=='Romania']

plt.scatter(data1['Date'], np.log(data1['ConfirmedCases']+1))

data1 = df_train[df_train['Country']=='Finland']

plt.scatter(data1['Date'], np.log(data1['ConfirmedCases']+1))

data1 = df_train[df_train['State']=='Hubei']

plt.scatter(data1['Date'], np.log(data1['ConfirmedCases']+1))
# convert the time to days, and combine state and country into one field

NULL_VAL = "NULL_VAL"



def fillState(state, country):

    if state == NULL_VAL: 

        return country

    

    return state



X_Train = df_train.loc[:, ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]

X_Train['State'].fillna(NULL_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

#convert to days

firstDay = np.min(X_Train['Date'])

X_Train.loc[:, 'Date'] = (X_Train['Date']-np.min(X_Train['Date'])).values / 86400000000000

X_Train["Date"]  = X_Train["Date"].astype(int)



X_Test = df_test.loc[:, ['State', 'Country', 'Date', 'ForecastId']]

X_Test['State'].fillna(NULL_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Test.loc[:, 'Date'] = (X_Test['Date']-firstDay).values / 86400000000000

X_Test["Date"]  = X_Test["Date"].astype(int)
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



countries = X_Train.Country.unique()



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in countries:

    states = X_Train.loc[X_Train.Country == country, :].State.unique()

    #print(country)

    for state in states:

        condition_train = (X_Train.Country == country) & (X_Train.State == state)

       

        # Get X and y (train)

        X_Train_CS  = X_Train.loc[condition_train, ['Date', 'ConfirmedCases', 'Fatalities']]

        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']

        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']

        X_Train_CS  = X_Train_CS.loc[:, ['Date']]



        # Get X and y (test)

        condition_test = (X_Test.Country == country) & (X_Test.State == state)

        X_Test_CS = X_Test.loc[condition_test, ['Date', 'ForecastId']]

        

        # Save forcast id for submission

        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']

        X_Test_CS    = X_Test_CS.loc[:, ['Date']]

        

        model1 = XGBRegressor(n_estimators=1000)

        model1.fit(X_Train_CS, y1_Train_CS)

        y1_pred = model1.predict(X_Test_CS)

        

        model2 = XGBRegressor(n_estimators=1000)

        model2.fit(X_Train_CS, y2_Train_CS)

        y2_pred = model2.predict(X_Test_CS)

        

        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

        df_out = pd.concat([df_out, df], axis=0)

    # Done for state loop

# Done for country Loop



df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()
df_out
df_out.to_csv('submission.csv', index=False)