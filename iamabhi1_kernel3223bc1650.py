import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as snp

import datetime as dt

%matplotlib inline
covid_train_data = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
covid_train_data.head(5)
covid_test_data = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
covid_test_data.head(5)
covid_train_data.describe()
covid_train_data = covid_train_data.dropna(axis=1)

covid_test_data = covid_test_data.dropna(axis=1)
covid_train_data[(covid_train_data['Country/Region'] == 'India') & covid_train_data['Fatalities'] != 0].plot(kind='line', x='ConfirmedCases', y='Fatalities')
covid_train_data[covid_train_data['ConfirmedCases'] == 67800]['Country/Region'].unique()
filtered_data = covid_train_data.groupby('Date').sum()
filtered_data.plot(kind='line', x='ConfirmedCases', y='Fatalities')
china_data = covid_train_data[covid_train_data['Country/Region'] == 'China'].groupby('Date').sum()
china_data.plot.hexbin(x='ConfirmedCases', y='Fatalities', gridsize=20, cmap='viridis')
# countries with having Fatalities > 10

fatal_data = covid_train_data[covid_train_data['Fatalities'] > 10].groupby('Country/Region').sum()
fatal_data.reset_index(inplace=True)
plt.figure(figsize=(15,10))

snp.barplot(y='Country/Region', x='ConfirmedCases', data=fatal_data)
plt.figure(figsize=(15,10))

snp.barplot(y='Country/Region', x='Fatalities', data=fatal_data)
covid_train_data['month'] = pd.DatetimeIndex(covid_train_data['Date']).month

covid_train_data.drop('Date', axis=1, inplace=True)

# -- TEST -- #

covid_test_data['month'] = pd.DatetimeIndex(covid_test_data['Date']).month

covid_test_data.drop('Date', axis=1, inplace=True)
X = covid_train_data.drop(['Country/Region', 'ConfirmedCases', 'Fatalities'], axis=1)

y1 = covid_train_data['ConfirmedCases']

y2 = covid_train_data['Fatalities']

x_test = covid_test_data.drop('Country/Region', axis=1)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X = scalar.fit_transform(X)

x_test = scalar.fit_transform(x_test)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y1.values)
prediction = model.predict(x_test)
predicted_data = pd.DataFrame(prediction)
predicted_data.columns = ['ConfirmedCases']
from sklearn.ensemble import RandomForestRegressor
new_model = RandomForestRegressor()

new_model.fit(X, y2.values)
fatalities_prediction = new_model.predict(x_test)
fatal_pred = pd.DataFrame(fatalities_prediction)
fatal_pred.columns = ['Fatalities']
submission_data = pd.concat([covid_test_data['ForecastId'], predicted_data, fatal_pred], axis=1)
submission_data['Fatalities'] = submission_data['Fatalities'].astype(int)

submission_data['ConfirmedCases'] = submission_data['ConfirmedCases'].astype(int)
submission_data.to_csv('submission.csv', index=False)
submission_data