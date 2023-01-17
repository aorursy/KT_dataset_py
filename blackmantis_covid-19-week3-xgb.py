import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from itertools import cycle, islice

import seaborn as sb

import matplotlib.dates as dates

from datetime import datetime

from sklearn import preprocessing

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
train.shape
test.shape
train.info()
train[['ConfirmedCases', 'Fatalities']].describe()
curr_date = train['Date'].max()

world_cum_confirmed = sum(train[train['Date'] == curr_date].ConfirmedCases)

world_cum_fatal = sum(train[train['Date'] == curr_date].Fatalities)

print('Number of Countires: ', len(train['Country_Region'].unique()))

print('End date in train dset: ', curr_date)

print('Number of confirmed cases: ', world_cum_confirmed)

print('Number of fatal cases: ', world_cum_fatal)
top_country_c = train[train['Date'] == curr_date].groupby(['Date','Country_Region']).sum().sort_values(['ConfirmedCases'], ascending=False)

top_country_c.head(20)
top_country_c.tail(20)
top_country_f = train[train['Date'] == curr_date].groupby(['Date','Country_Region']).sum().sort_values(['Fatalities'], ascending=False)

top_country_f.head(20)
train['MortalityRate'] = train['Fatalities'] / train['ConfirmedCases']

train['MortalityRate'] = train['MortalityRate'].fillna(0.0)
train['MortalityRate']
train['Date'] = pd.to_datetime(train['Date'])

train_data_by_date = train.groupby(['Date'],as_index=True).agg({'ConfirmedCases': 'sum','Fatalities': 'sum', 'MortalityRate':'mean'})



train_data_by_date.MortalityRate.plot(figsize=(15,10),x_compat=True, legend='Mortality Rate',color='r')
top_country_m = train[train['Date'] == curr_date].groupby(['Country_Region']).sum().sort_values(['MortalityRate'], ascending=False)

top_country_m.head(10)
top_country_m.MortalityRate.head(10).plot(figsize=(15,10),kind='barh')

plt.xlabel("Mortality Rate")

plt.title("First 10 Countries with Highest Mortality Rate")
for df in [test]:

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
missed = "NA"



def State(state, country):

    if state == missed: return country

    return state
for df in [train, test]:

    df['Province_State'].fillna(missed, inplace=True)

    df['Province_State'] = df.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : State(x['Province_State'], x['Country_Region']), axis=1)



    df.loc[:, 'Date'] = df.Date.dt.strftime("%m%d")

    df["Date"]  = df["Date"].astype(int)
train.head()
test.head(5)
label_encoder = preprocessing.LabelEncoder()



for df in [train, test]:

    df['Country_Region'] = label_encoder.fit_transform(df['Country_Region'])

    df['Province_State'] = label_encoder.fit_transform(df['Province_State'])
def RF():

    model = RandomForestRegressor(n_estimators = 100) 

    return model



def XGB():

    model = XGBRegressor(n_estimators=1300)

    return model



def LGBM():

    model = LGBMRegressor(iterations=2)

    return model
unique_countries = train['Country_Region'].unique()
sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



for country in range(len(unique_countries)):

    current_country_train = train.loc[train['Country_Region'] == unique_countries[country]]

    current_country_test = test.loc[test['Country_Region'] == unique_countries[country]]    

    

    features = ['Country_Region', 'Province_State', 'Date']

    X_train = current_country_train[features].to_numpy()

    y1_train = current_country_train[['ConfirmedCases']].to_numpy()

    y2_train = current_country_train[['Fatalities']].to_numpy()

    X_test = current_country_test[features].to_numpy()

    

    y1_train = y1_train.reshape(-1)

    y2_train = y2_train.reshape(-1)



    

    

    model1 = XGB()

    model1.fit(X_train, y1_train)

    res_cnf_cls = np.round(model1.predict(X_test))



    model2 = XGB()

    model2.fit(X_train, y2_train)

    res_fac = np.round(model2.predict(X_test))



    current_country_test_Id = current_country_test.loc[:, 'ForecastId']

    pred = pd.DataFrame({'ForecastId': current_country_test_Id, 'ConfirmedCases': res_cnf_cls, 'Fatalities': res_fac})

    

    sub = pd.concat([sub, pred], axis=0)
sub.ForecastId = sub.ForecastId.astype('int')



sub.to_csv('submission.csv', index=False)

sub