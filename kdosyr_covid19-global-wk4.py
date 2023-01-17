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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
# SIMPLEST SUBMISSION



# merged = pd.merge(test, train, on=["Province_State","Country_Region", "Date"], how="left")

# m = pd.DataFrame(merged.groupby(['ForecastId'])['ConfirmedCases', 'Fatalities'].sum())

# m.reset_index(inplace=True)

# sub = m.copy()

# sub.to_csv("submission.csv", index=False)
# FILL IN COUNTRY IF NO STATE



def use_country(state, country):

    if pd.isna(state):

        return country

    else:

        return state



train['Province_State'] = train.apply(lambda x: use_country(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'] = test.apply(lambda x: use_country(x['Province_State'], x['Country_Region']), axis=1)



train['Province_State'].fillna('', inplace=True)

test['Province_State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])
from xgboost import XGBRegressor 



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



# for country in list(set(train['Country_Region']))[:2]:

for country in set(train['Country_Region']):

  country_df = train[train['Country_Region'] == country]

  for state in set(country_df['Province_State']):

    df = country_df[country_df['Province_State'] == state]

    

    df['Date'] = df['Date'].dt.strftime("%m%d")

    df['Date'] = df['Date'].astype(int)

  

    test_df= test[(test['Country_Region'] == country) & (test['Province_State'] == state)]

    X_Test_CS_Id = test_df['ForecastId']





    # print('****************',X_Test_CS_Id)

    test_df['Date'] = test_df['Date'].dt.strftime("%m%d")

    test_df['Date'] = test_df['Date'].astype(int)





    X_Train_CS = df[['Country_Region', 'Province_State', 'Date']]

    

    X_Train_CS['Country_Region'] = 0

    X_Train_CS['Province_State'] = 0



    X_Test_CS = test_df[['Country_Region', 'Province_State', 'Date']]

    X_Test_CS['Country_Region'] = 0

    X_Test_CS['Province_State'] = 0



    y1_Train_CS = df['ConfirmedCases']

    y2_Train_CS = df['Fatalities']

    

    model1 = XGBRegressor(n_estimators=1000)

    model1.fit(X_Train_CS, y1_Train_CS)

    y1_pred = model1.predict(X_Test_CS)

    # print(y1_pred)



    model2 = XGBRegressor(n_estimators=1000)

    model2.fit(X_Train_CS, y2_Train_CS)

    y2_pred = model2.predict(X_Test_CS)



    df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})

    df_out = pd.concat([df_out, df], axis=0)
df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.tail()



df_out.to_csv('submission.csv', index=False)