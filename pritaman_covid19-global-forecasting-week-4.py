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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeClassifier
from fastai import *

from fastai.tabular import *

from fastai.callbacks import *
path1 = "/kaggle/input/covid19-global-forecasting-week-4"

path2 = "/kaggle/input/covid19-demographic-predictors"

path3 = "/kaggle/input/covid19-country-data-wk3-release"

path4 = "/kaggle/input/countryinfo"
train_df = pd.read_csv(f"{path1}/train.csv", parse_dates=['Date'])

test_df = pd.read_csv(f"{path1}/test.csv", parse_dates=['Date'])
add_datepart(train_df, 'Date', drop=False)
add_datepart(test_df, 'Date', drop=False)
missed = "NA"



def State(state, country):

    if state == missed: return country

    return state
metadata_df = pd.read_csv(f"{path3}/Data Join - RELEASE.csv", thousands=',')
country_df = pd.read_csv(f"{path4}/covid19countryinfo.csv",thousands=",", parse_dates=['quarantine', 'schools', 'publicplace', 'gathering', 'nonessential'])
testinfo = pd.read_csv(f'{path4}/covid19tests.csv', thousands=",")
country_df.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)

testinfo.rename(columns={'region': 'Province_State', 'country': 'Country_Region'}, inplace=True)

testinfo = testinfo.drop(['alpha3code', 'alpha2code', 'date'], axis=1)
group_cols = ['Country_Region', 'Province_State']
group = train_df[train_df["Fatalities"] > 1].groupby(group_cols)

res_df = (group.Fatalities.last() / group.ConfirmedCases.last()).reset_index()

fatalities = res_df.rename(columns={0 : 'FatalityRate'})
group = train_df[train_df['ConfirmedCases'] >= 1].groupby(group_cols)

first_confirmed = group.Dayofyear.first().reset_index().rename(columns={'Dayofyear': "First_Confirmed"})
group = train_df[train_df['ConfirmedCases'] >= 50].groupby(group_cols)

first_50= group.Dayofyear.first().reset_index().rename(columns={'Dayofyear': "First_50"})
group = train_df[train_df['ConfirmedCases'] >= 100].groupby(group_cols)

first_hundred = group.Dayofyear.first().reset_index().rename(columns={'Dayofyear': "First_Hundred"})
train_df = pd.merge(train_df, metadata_df, how='left')

train_df = pd.merge(train_df, country_df, how='left')

train_df = pd.merge(train_df, testinfo, how='left', left_on=group_cols, right_on=group_cols)
train_df = pd.merge(train_df, fatalities, how='left')

train_df = pd.merge(train_df, first_confirmed, how='left')

train_df = pd.merge(train_df, first_50, how='left')

train_df = pd.merge(train_df, first_hundred, how='left')
test_df = pd.merge(test_df, metadata_df, how='left')

test_df = pd.merge(test_df, country_df, how='left')

test_df = pd.merge(test_df, testinfo, how='left', left_on=group_cols, right_on=group_cols)

test_df = pd.merge(test_df, fatalities, how='left')

test_df = pd.merge(test_df, first_confirmed, how='left')

test_df = pd.merge(test_df, first_50, how='left')

test_df = pd.merge(test_df, first_hundred, how='left')
for df in [train_df, test_df]:

    df['Province_State'].fillna(missed, inplace=True)

    df['Province_State'] = df.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : State(x['Province_State'], x['Country_Region']), axis=1)



    df.loc[:, 'Date'] = df.Date.dt.strftime("%m%d")

    df["Date"]  = df["Date"].astype(int)
le = preprocessing.LabelEncoder()



for df in [train_df, test_df]:

    df['Country_Region'] = le.fit_transform(df['Country_Region'])

    df['Province_State'] = le.fit_transform(df['Province_State'])
countries = train_df['Country_Region'].unique()
sub = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

features = ['Country_Region', 'Province_State', 'Date']

for country in range(len(countries)):

    country_train = train_df.loc[train_df['Country_Region'] == countries[country]]

    country_test = test_df.loc[test_df['Country_Region'] == countries[country]]

    

    states = train_df.loc[train_df['Country_Region'] == countries[country],:].Province_State.unique()

    for state in states:

        #X_train = train_df.loc[train_df['Country_Region'] == countries[country] & train_df['Province_State'] == state, features].to_numpy()

        #y1_train = train_df.loc[train_df['Country_Region'] == countries[country] & train_df['Province_State'] == state, ['ConfirmedCases']].to_numpy()

        #y2_train = train_df.loc[train_df['Country_Region'] == countries[country] & train_df['Province_State'] == state, ['Fatalities']].to_numpy()

        

        train = country_train.loc[country_train['Province_State'] == state]

        X_train = train[features].to_numpy()

        y1_train = train[['ConfirmedCases']].to_numpy()

        y2_train = train[['Fatalities']].to_numpy()

        

        test = country_test.loc[country_test['Province_State'] == state]

        X_test = test[features].to_numpy()

        #X_test = test_df.loc[test_df['Country_Region'] == countries[country] & test_df['Province_State'] == state]

        

        #model1 = XGBRegressor(n_estimators=1000)

        model1 = DecisionTreeClassifier()

        model1.fit(X_train, y1_train)

        pred1 = np.round(model1.predict(X_test))

        

        #model2 = XGBRegressor(n_estimators=1000)

        model2 = DecisionTreeClassifier()

        model2.fit(X_train, y2_train)

        pred2 = np.round(model2.predict(X_test))

        

        country_test_Id = test.loc[:, 'ForecastId']

        pred = pd.DataFrame({'ForecastId': country_test_Id, 'ConfirmedCases': pred1, 'Fatalities': pred2})

        sub = pd.concat([sub, pred], axis=0)
sub.ForecastId = sub.ForecastId.astype('int')

sub.to_csv('submission.csv', index=False)