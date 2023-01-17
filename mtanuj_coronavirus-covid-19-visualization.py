import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVR

%matplotlib inline 
confirmed_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Confirmed.csv')

deaths_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Deaths.csv')

recoveries_df = pd.read_csv('../input/corona-virus-report/time_series_2019-ncov-Recovered.csv')
confirmed_df.head()
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
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
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
kernel = ['linear', 'rbf', 'poly', 'sigmoid']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

degree = [2, 3, 4, 5]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking, 'degree' : degree}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)

svm_search.fit(days_since_1_22, world_cases)
svm_confirmed = svm_search.best_estimator_

pred = svm_confirmed.predict(days_since_1_22)
plt.figure(figsize=(20, 12))

plt.plot(dates, world_cases)

plt.plot(dates, pred, linestyle='dashed')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(dates, total_deaths, color='red')

plt.title('# of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Deaths', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(20, 12))

plt.plot(dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])

plt.xlabel('Time', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(dates, total_recovered, color='green')

plt.title('# of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(dates, total_deaths, color='red')

plt.plot(dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(total_recovered, total_deaths)

plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)

plt.xlabel('# of Coronavirus Recoveries', size=30)

plt.ylabel('# of Coronavirus Deaths', size=30)

plt.xticks(size=15)

plt.show()
latest_confirmed = confirmed_df[dates[-1]]

latest_deaths = deaths_df[dates[-1]]

latest_recoveries = recoveries_df[dates[-1]]
unique_provinces =  confirmed_df['Province/State'].unique()

unique_provinces
province_confirmed_cases = []

for i in unique_provinces:

    province_confirmed_cases.append(latest_confirmed[confirmed_df['Province/State']==i].sum())
unique_countries =  confirmed_df['Country/Region'].unique()

unique_countries 
country_confirmed_cases = []

for i in unique_countries:

    country_confirmed_cases.append(latest_confirmed[confirmed_df['Country/Region']==i].sum())
# number of cases per country/region

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
# number of cases per province/state/city



for i in range(len(unique_provinces)):

    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')
nan_indices = [] 



# handle nan if there is any, it is usually a float: float('nan')



for i in range(len(unique_provinces)):

    if type(unique_provinces[i]) == float:

        nan_indices.append(i)



unique_provinces = list(unique_provinces)

province_confirmed_cases = list(province_confirmed_cases)



for i in nan_indices:

    unique_provinces.pop(i)

    province_confirmed_cases.pop(i)
plt.figure(figsize=(10, 10))

plt.barh(unique_countries, country_confirmed_cases)

plt.title('# of Coronavirus Confirmed Cases in Countries/Regions')

plt.xlabel('# of Covid19 Confirmed Cases')

plt.show()
# lets look at it in a logarithmic scale 

log_country_confirmed_cases = [math.log10(i) for i in country_confirmed_cases]

plt.figure(figsize=(10, 10))

plt.barh(unique_countries, log_country_confirmed_cases)

plt.title('Common Log # of Coronavirus Confirmed Cases in Countries/Regions')

plt.xlabel('Log of # of Covid19 Confirmed Cases')

plt.show()
plt.figure(figsize=(20, 15))

plt.barh(unique_provinces, province_confirmed_cases)

plt.title('# of Coronavirus Confirmed Cases in Provinces/States')

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.pie(country_confirmed_cases, colors=c)

plt.legend(unique_countries, loc='best')

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.pie(province_confirmed_cases, colors=c)

plt.legend(unique_provinces, loc='best')

plt.show()