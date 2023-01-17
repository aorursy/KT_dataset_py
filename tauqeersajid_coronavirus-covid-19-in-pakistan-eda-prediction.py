import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn')

%matplotlib inline 
# confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

# deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

# recoveries_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')



confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
pakistan_confirmed_df = confirmed_df[confirmed_df['Country/Region'] == 'Pakistan']
pakistan_deaths_df = deaths_df[deaths_df['Country/Region'] == 'Pakistan']
pakistan_recoveries_df = recoveries_df[recoveries_df['Country/Region'] == 'Pakistan']
pakistan_confirmed_df
confirmed = pakistan_confirmed_df.iloc[:, 4:].T

deaths = pakistan_deaths_df.iloc[:, 4:].T

recoveries = pakistan_recoveries_df.iloc[:, 4:].T
confirmed = confirmed.rename(columns={177: "confirmed_cases"})

deaths = deaths.rename(columns={177: "deaths"})

recoveries = recoveries.rename(columns={174: "recoveries"})

confirmed.index = pd.to_datetime(confirmed.index)

deaths.index = pd.to_datetime(deaths.index)

recoveries.index = pd.to_datetime(recoveries.index)
print('Pakistan Confirmed Cases ' + str(confirmed['confirmed_cases'][-1]))

print('Pakistan Death Cases ' + str(deaths['deaths'][-1]))

print('Pakistan Recovery Cases ' + str(recoveries['recoveries'][-1]))

pakistan_active_cases = (confirmed['confirmed_cases'] - deaths['deaths'] - recoveries['recoveries'])

print('Active cases in Pakistan ' + str(pakistan_active_cases[-1]))
print(confirmed.index)

print(deaths.index)

print(recoveries.index)
dates = confirmed.index

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

pakistan_cases = confirmed['confirmed_cases'].T

pakistan_total_deaths = deaths['deaths'].T

pakistan_total_recovered = recoveries['recoveries'].T
# calculate rates

mortality_rate = []

recovery_rate = [] 

mortality_rate.append(deaths['deaths']/confirmed['confirmed_cases'])

recovery_rate.append(recoveries['recoveries']/confirmed['confirmed_cases'])
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
# X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, confirmed['confirmed_cases'], test_size=0.01, shuffle=False) 
confirmed_cases = np.array(confirmed['confirmed_cases']).reshape(-1, 1)
poly = PolynomialFeatures(degree=4)

poly_X_train_confirmed = poly.fit_transform(days_since_1_22)

poly_X_test_confirmed = poly.fit_transform(confirmed_cases)

poly_future_forcast = poly.fit_transform(future_forcast)
linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train_confirmed, confirmed_cases)

#test_linear_pred = linear_model.predict(poly_X_test_confirmed)

linear_pred = linear_model.predict(poly_future_forcast)

# print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

# print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)
plt.figure(figsize=(15, 8))

plt.scatter(adjusted_dates, pakistan_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

#plt.savefig('Pakistan_cases_over_time_scatter_plot.png')

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

#plt.savefig('Pakistan_cases_over_time_line_plot.png')

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_cases)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('Pakistan # of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.savefig('Pakistan_cases_next_10_day_pred.png')

plt.show()
# Future predictions using Polynomial Regression 

linear_pred = linear_pred.reshape(1,-1)[0]

print('Polynomial regression future predictions:')

set(zip(future_forcast_dates[-10:], np.round(linear_pred[-10:])))
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_active_cases, color='purple')

plt.title('# of Coronavirus Active Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Active Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_total_deaths, color='red')

plt.title('# of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
nan = np.isnan(mortality_rate)

mortality_rate = np.array(mortality_rate)

mortality_rate[nan] = 0.0
mean_mortality_rate = np.mean(mortality_rate.T)

plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, mortality_rate.T, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_total_recovered, color='green')

plt.title('# of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
nan = np.isnan(recovery_rate)

recovery_rate = np.array(recovery_rate)

recovery_rate[nan] = 0.0
mean_recovery_rate = np.mean(recovery_rate.T)

plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, recovery_rate.T, color='blue')

plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')

plt.title('Recovery Rate of Coronavirus Over Time', size=30)

plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Recovery Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(adjusted_dates, pakistan_total_deaths, color='r')

plt.plot(adjusted_dates, pakistan_total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('# Death and Recoveries of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.savefig('Pakistan_Death_Recoveries_cases_over_time.png')

plt.show()
plt.figure(figsize=(15, 8))

plt.plot(pakistan_total_recovered, pakistan_total_deaths)

plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)

plt.xlabel('# of Coronavirus Recoveries', size=30)

plt.ylabel('# of Coronavirus Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()