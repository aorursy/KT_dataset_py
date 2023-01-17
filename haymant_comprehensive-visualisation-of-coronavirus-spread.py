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

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 

plt.style.use('seaborn')

%matplotlib inline
confirmed_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recoveries_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
confirmed_df.head()
cols = confirmed_df.keys()

dates =[x for x in cols if x not in ["Province/State","Country/Region","Lat","Long"]]
confirmed_df['TotalConfirmedCases']=0

for date in dates:

    confirmed_df['TotalConfirmedCases']=confirmed_df['TotalConfirmedCases']+confirmed_df[date]
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
world_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 

china_cases = [] 

italy_cases = []

us_cases = []

india_cases = []



for i in dates:

    confirmed_sum = confirmed[i].sum()

    death_sum = deaths[i].sum()

    recovered_sum = recoveries[i].sum()

    

    # confirmed, deaths, recovered, and active

    world_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)

    

    # calculate rates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)

    

    # case studies 

    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())

    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())

    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())

    india_cases.append(confirmed_df[confirmed_df['Country/Region']=='India'][i].sum())
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)
poly = PolynomialFeatures(degree=7)

poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

poly_future_forcast = poly.fit_transform(future_forcast)



# polynomial regression

linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(poly_X_test_confirmed)

linear_pred = linear_model.predict(poly_future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Polynomial Regression Prediction'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, china_cases)

plt.plot(adjusted_dates, italy_cases)

plt.plot(adjusted_dates, us_cases)

plt.plot(adjusted_dates, india_cases)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['China', 'Italy', 'US','India'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
# Future predictions using Polynomial Regression 

linear_pred = linear_pred.reshape(1,-1)[0]

print('Polynomial regression future predictions:')

set(zip(future_forcast_dates[-10:], np.round(linear_pred[-10:])))
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_active, color='purple')

plt.title('# of Coronavirus Active Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Active Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.title('# of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_recovered, color='green')

plt.title('# of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
mean_recovery_rate = np.mean(recovery_rate)

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, recovery_rate, color='blue')

plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')

plt.title('Recovery Rate of Coronavirus Over Time', size=30)

plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('Recovery Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(total_recovered, total_deaths)

plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)

plt.xlabel('# of Coronavirus Recoveries', size=30)

plt.ylabel('# of Coronavirus Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
latest_confirmed = confirmed_df[dates[-1]]

latest_deaths = deaths_df[dates[-1]]

latest_recoveries = recoveries_df[dates[-1]]
unique_countries =  list(confirmed_df['Country/Region'].unique())

country_confirmed_cases = []

no_cases = []

for i in unique_countries:

    cases = latest_confirmed[confirmed_df['Country/Region']==i].sum()

    if cases > 0:

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

        

for i in no_cases:

    unique_countries.remove(i)

    

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i] = latest_confirmed[confirmed_df['Country/Region']==unique_countries[i]].sum()
# number of cases per country/region

print('Confirmed Cases by Countries/Regions:')

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category

# As per latest data available.

visual_unique_countries = [] 

visual_confirmed_cases = []

others = np.sum(country_confirmed_cases[10:])



for i in range(len(country_confirmed_cases[:10])):

    visual_unique_countries.append(unique_countries[i])

    visual_confirmed_cases.append(country_confirmed_cases[i])

    

visual_unique_countries.append('Others')

visual_confirmed_cases.append(others)
plt.figure(figsize=(16, 9))

plt.barh(visual_unique_countries, visual_confirmed_cases)

plt.title('# of Covid-19 Confirmed Cases in Countries/Regions', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per Country', size=20)

plt.pie(visual_confirmed_cases, colors=c)

plt.legend(visual_unique_countries, loc='best', fontsize=15)

plt.show()