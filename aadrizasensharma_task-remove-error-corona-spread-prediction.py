# Importing all the important libraries.



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import random

import math

import time

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,mean_absolute_error

import datetime

import operator

import numpy as np

from sklearn.linear_model import LinearRegression

plt.style.use('seaborn')

%matplotlib inline
# loading all the three datasets



confirmed_cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
deaths_reported = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
recovered_cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
# display the head of the Dataset



confirmed_cases.head()
deaths_reported.head()
recovered_cases.head()
# Extracting all the columns using the .keys() function.

cols = confirmed_cases.keys()

cols
# Extracting only the dates columns that have information of confirmed,deaths and recovered cases.

confirmed =  confirmed_cases.loc[:, cols[4]:cols[-1]]
deaths = deaths_reported.loc[:, cols[4]:cols[-1]]
recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]
# Check the head of the outbreak cases.

confirmed.head()
# Finding the total confirmed cases,death cases and the recovered cases and append them to an 4 empty lists.

# Also, calculate the total mortality rate which is the death sum/confirmed cases.



dates = confirmed.keys()

world_cases = []

total_deaths = []

mortality_rate = []

total_recovered = []



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    mortality_rate.append(death_sum/confirmed_sum)

    total_recovered.append(recovered_sum)
# Lets display each of the newly created variables

confirmed_sum
death_sum
recovered_sum
world_cases
# Convert all the dates and the cases in the form of a numpy array



days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1,1)

total_deaths = np.array(total_deaths).reshape(-1,1)

total_recovered = np.array(total_recovered).reshape(-1,1)
days_since_1_22
world_cases
total_deaths
total_recovered
# Future forecasting for the next 10 days



days_in_future = 10

future_forecast = np.array([i for i in range (len(dates)+days_in_future)]).reshape(-1,1)

adjusted_dates = future_forecast[:-10]
future_forecast
# Convert all the integers into datetime for better visualization.

start = '1/22/2020'

start_date = datetime.datetime.strptime(start,'%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forecast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
# For visualization with the latest data of the 15th of march



latest_confirmed = confirmed_cases[dates[-1]]

latest_deaths = deaths_reported[dates[-1]]
latest_confirmed
latest_deaths
# Find the list of unique countries

unique_countries = list(confirmed_cases['Country/Region'].unique())

unique_countries
# The next line of code will basically calculate the total number of confirmed cases by each country.



country_confirmed_cases = []

no_cases = []

for i in unique_countries:

    cases = latest_confirmed[confirmed_cases['Country/Region']==i].sum()

    if cases > 0:

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

        

for i in no_cases:

    unique_countries.remove(i)

    

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i] = latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()
# Number of cases per Country/Region

print('Confirmed Cases by Countries/Regions:')

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
# Find the list of unique provinces

unique_provinces =  list(confirmed_cases['Province/State'].unique())

# those are countries, which are not provinces/states.

outliers = ['United Kingdom', 'Denmark', 'France']

for i in outliers:

    unique_provinces.remove(i)
# Finding the number of confirmed cases per provinces,state or city.

province_confirmed_cases = []

no_cases = [] 

for i in unique_provinces:

    cases = latest_confirmed[confirmed_cases['Province/State']==i].sum()

    if cases > 0:

        province_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

# remove areas with no confirmed cases

for i in no_cases:

    unique_provinces.remove(i)
# Number of cases per province/state/city

for i in range(len(unique_provinces)):

    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')
# Handling nan values if there is any,it is usually a float: float('nan')



nan_indices = [] 



for i in range(len(unique_provinces)):

    if type(unique_provinces[i]) == float:

        nan_indices.append(i)



unique_provinces = list(unique_provinces)

province_confirmed_cases = list(province_confirmed_cases)



for i in nan_indices:

    unique_provinces.pop(i)

    province_confirmed_cases.pop(i)
# Plot a bar graph to see the total confirmed cases across different countries.

plt.figure(figsize=(32,32))

plt.barh(unique_countries,country_confirmed_cases)

plt.title('Number of Covid-19 confirmed cases in Countries')

plt.xlabel('Number of Covid19 Confirmed cases')

plt.show()
# Plot a bar graph to see the total confirmed cases between mainland china and outside mainland china.

china_confirmed = latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()

outside_mainland_china_confirmed = np.sum(country_confirmed_cases) - china_confirmed

plt.figure(figsize=(16, 9))

plt.barh('Mainland China', china_confirmed)

plt.barh('Outside Mainland China', outside_mainland_china_confirmed)

plt.title('Number of  Confirmed Coronavirus Cases')

plt.show()
# Print the total cases in mainland in China and outside of it.

print('Outside Mainland China {} cases:'.format(outside_mainland_china_confirmed))

print('Mainland China: {} cases'.format(china_confirmed))

print('Total: {} cases'.format(china_confirmed+outside_mainland_china_confirmed))
# Only show 10 countries with the most confirmed cases, the rest are grouped into the  category named others

visual_unique_countries = [] 

visual_confirmed_cases = []

others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):

    visual_unique_countries.append(unique_countries[i])

    visual_confirmed_cases.append(country_confirmed_cases[i])



visual_unique_countries.append('Others')

visual_confirmed_cases.append(others)
# Visualize the 10 Countries

plt.figure(figsize=(32, 18))

plt.barh(visual_unique_countries, visual_confirmed_cases)

plt.title('Number of Covid-19 Confirmed Cases in Countries/Regions', size=20)

plt.show()
# Create a Pie chart to see the total confirmed cases in 10 different countries.



c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per Country')

plt.pie(visual_confirmed_cases, colors=c)

plt.legend(visual_unique_countries, loc='best')

plt.show()
# Create a pie chart to see the total confirmed cases in 10 different countries outside China

c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in Countries Outside of Mainland China')

plt.pie(visual_confirmed_cases[1:], colors=c)

plt.legend(visual_unique_countries[1:], loc='best')

plt.show()
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)
# Building the SVM Model

kernel = ['poly','sigmoid','rbf']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking= [True, False]

svm_grid = {'kernel': kernel,'C': c,'gamma': gamma,'epsilon':epsilon,'shrinking':shrinking}

svm = SVR(kernel='poly')

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_search.best_params_
svm_confirmed = svm_search.best_estimator_

svm_pred = svm_confirmed.predict(future_forecast)
svm_confirmed
svm_pred
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# Total number of coronavirus cases over time

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.title('Number of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Confirmed vs Predicted Cases

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forecast, svm_pred, linestyle='dashed', color='purple')

plt.title('Number of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Number of Cases', size=30)

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Prediction for the next 10 days using SVM

print('SVM future predictions:')

set(zip(future_forcast_dates[-10:], svm_pred[-10:]))
# Using Linear Regression Model to make Predictions

linear_model = LinearRegression(normalize = True,fit_intercept = True)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forecast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
plt.plot(y_test_confirmed)

plt.plot(test_linear_pred)
# Graphing the number of confirmed cases, deaths, active cases, and the mortality rate over time, as well as the number of recoveries

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forecast, linear_pred, linestyle='dashed', color='orange')

plt.title('Number of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Number of Cases', size=30)

plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Prediction for the next 10 days using Linear Regression

print('Linear regression future predictions:')

print(linear_pred[-10:])
# Total Deaths over time

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.title('Number of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Number of Deaths', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])

plt.xlabel('Time', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Coronavirus cases Recovered over Time



plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_recovered, color='green')

plt.title('Number of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Number of Coronavirus cases recovered vs the number of deaths

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('Number of Coronavirus Cases', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Coronavirus Deaths vs Recoveries

plt.figure(figsize=(20, 12))

plt.plot(total_recovered, total_deaths)

plt.title('Number of Coronavirus Deaths vs. Number of Coronavirus Recoveries', size=30)

plt.xlabel('Total Number of Coronavirus Recoveries', size=30)

plt.ylabel('Total Number of Coronavirus Deaths', size=30)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()