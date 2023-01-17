import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv", parse_dates=['Date'])

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv", parse_dates=['Date'])
display(train.head(5))

display(train.describe())

print(train.dtypes)

print("\n")

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())
display(test.head(5))

display(test.describe())

print("Number of Country_Region: ", test['Country_Region'].nunique())

print("Dates go from day", max(test['Date']), "to day", min(test['Date']), ", a total of", test['Date'].nunique(), "days")

print("Countries with Province/State informed: ", test[test['Province_State'].isna()==False]['Country_Region'].unique())
display(submission.head(5))

display(submission.describe())
train['geo_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)

train.head()
test['geo_id'] = test['Country_Region'].astype(str) + '_' + test['Province_State'].astype(str)

test.head()
fig, ax = plt.subplots()

train.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)
#train.groupby('geo_id').plot.line(x='Date', y='ConfirmedCases')
train[train.Country_Region == "China"].Province_State.unique()
train[train.Country_Region == "China"].groupby("Province_State").sum()
train_hubei = train[train['geo_id'] == "China_Hubei"]

train_hubei
fig, ax = plt.subplots()

train_hubei.plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)
area_features = train[['geo_id', 'Country_Region', 'Province_State']].drop_duplicates().set_index('geo_id')

area_features
countryinfo = pd.read_csv("../input/countryinfo/covid19countryinfo.csv", thousands=',')
print(countryinfo.dtypes)

display(countryinfo.head(5))

display(countryinfo.describe())
# extract population country data

pop_data = countryinfo[['country', "pop"]].rename(columns={'pop': 'pop_country'}).groupby("country").max()

pop_data
# left join pop_country to area_features

area_features = area_features.join(pop_data, how='left', on="Country_Region")

area_features
print("Number of countries with population: ", area_features[area_features['pop_country'].isna()==False].Country_Region.nunique())

print("Number of countries without population: ", area_features[area_features['pop_country'].isna()==True].Country_Region.nunique())

print("Countries without population: ", area_features[area_features['pop_country'].isna()==True].Country_Region.unique())
# fill country population (pop_country) NA with 100 000

area_features['pop_country'] = area_features['pop_country'].fillna(100000)
print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())
# add column num_states per country

num_states = area_features[['Country_Region', 'Province_State']].fillna("").groupby('Country_Region').count().rename(columns={'Province_State': "num_states"})

area_features = area_features.join(num_states, on="Country_Region")

area_features
# fill province_state population (pop) with pop_country / num_states

area_features['pop'] = area_features['pop_country'] / area_features['num_states']

area_features
date_of_first_infection = train[train['ConfirmedCases'] > 0].groupby(['geo_id']).agg({'Date': 'min'}).rename(columns={'Date': 'date_of_first_infection'})

date_of_first_infection
area_features = area_features.join(date_of_first_infection, on="geo_id")

area_features
hubei_curve = train[(train['Country_Region'] == 'China') & (train['Province_State'] == 'Hubei')]

hubei_curve = hubei_curve[['Date', 'ConfirmedCases', 'Fatalities']].set_index('Date')

hubei_curve
hubei_curve.plot.line()
import datetime

date_start_hubei = datetime.datetime(2019, 12, 15) #hubei_curve.index.min()

date_start_hubei
area_features['date_delta_hubei'] = area_features['date_of_first_infection'] - date_start_hubei

area_features
population_hubei = 58.5 * 10**6

population_hubei
area_features['pop_scale_hubei'] = area_features['pop'] / population_hubei

area_features
area_features
data = train[['geo_id', 'Date']]

data
# add Hubei Curve to all Areas

data = data.join(hubei_curve, on='Date')

data
data = data.join(area_features[['date_delta_hubei']], on="geo_id")

data
# translate by date_delta_hubei

data['Date'] = data['Date'] + data['date_delta_hubei']

data
data = data.join(area_features[['pop_scale_hubei']], on="geo_id")

data
# scale by pop_scale_hubei

data['ConfirmedCases'] = data['ConfirmedCases'] * data['pop_scale_hubei']

data['Fatalities'] = data['Fatalities'] * data['pop_scale_hubei']

data
# drop unneeded columns

data = data[['geo_id', 'Date', 'ConfirmedCases', 'Fatalities']]

data
fig, ax = plt.subplots()

data.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)
test
# join 

submission = pd.merge(test, data, how="left", on=["geo_id", 'Date'])

submission
submission = submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission
submission.fillna(0, inplace=True)

submission
submission.describe()
submission.to_csv('submission.csv', index=False)