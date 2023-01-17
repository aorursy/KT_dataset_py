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
import pandas as pd

from datetime import datetime

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

# from sklearn.linear_model import LogisticRegression

# from sklearn.linear_model import LinearRegression

# from sklearn.ensemble import RandomForestRegressor 
path = '/kaggle/input/covid19-global-forecasting-week-2/'

df_train = pd.read_csv(path+'train.csv')

df_test = pd.read_csv(path+'test.csv')
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
df_train.head()
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



df_train.loc[:, 'Date'] = df_train.Date.dt.strftime("%m%d")

df_train["Date"]  = df_train["Date"].astype(int)



print(df_train.shape)

df_train.head()
df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)

df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



df_test.loc[:, 'Date'] = df_test.Date.dt.strftime("%m%d")

df_test["Date"]  = df_test["Date"].astype(int)



print(df_test.shape)

df_test.head()
le = preprocessing.LabelEncoder()



df_train['Country_Region'] = le.fit_transform(df_train['Country_Region'])

df_train['Province_State'] = le.fit_transform(df_train['Province_State'])



df_test['Country_Region'] = le.fit_transform(df_test['Country_Region'])

df_test['Province_State'] = le.fit_transform(df_test['Province_State'])
df_train.head(2)
df_test.head(2)
unique_countries = df_train['Country_Region'].unique()
def build_model_1():

    model = RandomForestRegressor(n_estimators = 100, random_state = 0) 

    return model



def build_model_2():

    model = XGBRegressor(n_estimators=1000)

    return model



def build_model_3():

    model = DecisionTreeRegressor(random_state=1)

    return model



def build_model_4():

    model = LogisticRegression()

    return model



def build_model_5():

    model = LinearRegression()

    return model



def build_model_6():

    model = LGBMRegressor(random_state=5)

    return model



def build_model_7():

    model = LGBMRegressor(iterations=2)

    return model

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for country in range(len(unique_countries)):

    current_country_df_train = df_train.loc[df_train['Country_Region'] == unique_countries[country]]

    current_country_df_test = df_test.loc[df_test['Country_Region'] == unique_countries[country]]

    print("Current Country: ", unique_countries[country])

    

#     print(current_country_df_train.head(2))

#     print(current_country_df_test.head(2))

    

    # Create X and Y

    X_train = current_country_df_train[['Country_Region', 'Province_State', 'Date']].to_numpy()

    Y1_train = current_country_df_train[['ConfirmedCases']].to_numpy()

    Y2_train = current_country_df_train[['Fatalities']].to_numpy()

    X_test = current_country_df_test[['Country_Region', 'Province_State', 'Date']].to_numpy()

    

    Y1_train = Y1_train.reshape(-1)

    Y2_train = Y2_train.reshape(-1)

#     print(X_train.shape, Y1_train.shape, Y2_train.shape)

#     print(X_train.shape, Y1_train.reshape(-1), Y2_train.shape)

    

    

    model1 = build_model_2()

    model1.fit(X_train, Y1_train)

    res_cnf_cls = model1.predict(X_test)

    

    

    model2 = build_model_2()

    model2.fit(X_train, Y2_train)

    res_fac = model2.predict(X_test)

    

    current_country_df_test_Id = current_country_df_test.loc[:, 'ForecastId']

    df_ans = pd.DataFrame({'ForecastId': current_country_df_test_Id, 'ConfirmedCases': res_cnf_cls, 'Fatalities': res_fac})

    df_out = pd.concat([df_out, df_ans], axis=0)

   

    

#     if country == 1:

#         break
df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.head(100)
df_out.to_csv('submission.csv', index=False)