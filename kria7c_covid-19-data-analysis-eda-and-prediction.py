import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

%matplotlib inline 
confirmed_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered_df = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
confirmed_df.head()
deaths_df.head()
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recovered_df.loc[:, cols[4]:cols[-1]]
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
confirmed.isna().sum()
days_in_future = 3

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

adjusted_dates = future_forcast_dates[:-3]
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.title('Number of COVID Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.title('Number of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Total of Deaths', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
mean_mortality_rate = np.mean(mortality_rate)

plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, mortality_rate, color='orange')

plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')

plt.title('Mortality Rate of Coronavirus Over Time', size=30)

plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)])

plt.xlabel('Time', size=30)

plt.ylabel('Mortality Rate', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_recovered, color='green')

plt.title('Number of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Numer of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('Number of Coronavirus Cases', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('Number of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(total_recovered, total_deaths)

plt.title('Number of Coronavirus Deaths vs. Number of Coronavirus Recoveries', size=30)

plt.xlabel('Number of Coronavirus Recoveries', size=30)

plt.ylabel('Number of Coronavirus Deaths', size=30)

plt.xticks(size=15)

plt.show()
latest_confirmed = confirmed_df[dates[-1]]

latest_deaths = deaths_df[dates[-1]]

latest_recoveries = recovered_df[dates[-1]]
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
# number of cases per country/region

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per Country')

plt.pie(country_confirmed_cases, colors=c)

plt.legend(unique_countries, loc='best')

plt.show()
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.1, shuffle=False) 
X_train_confirmed.shape, y_train_confirmed.shape, X_test_confirmed.shape, y_test_confirmed.shape
kernel = ['linear', 'rbf']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)

svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_search.best_params_
svm_confirmed = svm_search.best_estimator_

svm_pred = svm_confirmed.predict(future_forcast)
rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 7, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"],}



# Instantiate RandomizedSearchCV model

rfr_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,

                                                    random_state=42),

                              param_distributions=rf_grid,

                              n_iter=30,

                              cv=5,

                              verbose=True)



# Fit the RandomizedSearchCV model

rfr_model.fit(X_train_confirmed, y_train_confirmed)

test_rfr_pred = rfr_model.predict(X_test_confirmed)

rfr_pred = rfr_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_rfr_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_rfr_pred, y_test_confirmed))
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast_dates, svm_pred, linestyle='dashed')

plt.title('Number of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('Number of Cases', size=30)

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast_dates, rfr_pred, linestyle='dashed')

plt.title('Number of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('Number of Cases', size=30)

plt.legend(['Confirmed Cases', 'Random Forest Regressor Predictions'])

plt.xticks(rotation=50, size=15)

plt.show()
print('Random Forest Regessor prediction: ', set(zip(future_forcast_dates[-4:], rfr_pred[-4:])))