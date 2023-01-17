import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

%matplotlib inline 
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
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
days_in_future = 5

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

adjusted_dates = future_forcast_dates[:-5]
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.2, shuffle=False) 
kernel = ['linear', 'rbf', 'poly', 'sigmoid']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

svm_search.fit(X_train_confirmed, y_train_confirmed)
svm_search.best_params_
svm_confirmed = svm_search.best_estimator_

svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
linear_model = LinearRegression(normalize=True)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
linear_model.coef_
plt.plot(y_test_confirmed)

plt.plot(test_linear_pred)
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast_dates, svm_pred, linestyle='dashed')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast_dates, linear_pred, linestyle='dashed', color='red')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time in Days', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])

plt.xticks(rotation=50, size=15)

plt.show()
# Future predictions using SVM 

print('SVM future predictions:')

set(zip(future_forcast_dates[-5:], svm_pred[-5:]))
# Future predictions using Linear Regression 

print('Linear regression future predictions:')

print(linear_pred[-5:])
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.title('# of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Deaths', size=30)

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

plt.title('# of Coronavirus Cases Recovered Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(rotation=50, size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

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
unique_provinces =  list(confirmed_df['Province/State'].unique())
province_confirmed_cases = []

no_cases = [] 

for i in unique_provinces:

    cases = latest_confirmed[confirmed_df['Province/State']==i].sum()

    if cases > 0:

        province_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

# remove areas with no confirmed cases

for i in no_cases:

    unique_provinces.remove(i)
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
plt.figure(figsize=(32, 18))

plt.barh(unique_countries, country_confirmed_cases)

plt.title('# of Covid-19 Confirmed Cases in Countries/Regions')

plt.xlabel('# of Covid19 Confirmed Cases')

plt.show()
outside_mainland_china_confirmed = np.sum(country_confirmed_cases[1:])

plt.figure(figsize=(16, 9))

plt.barh(unique_countries[0], country_confirmed_cases[0])

plt.barh('Outside Mainland China', outside_mainland_china_confirmed)

plt.title('# of Coronavirus Confirmed Cases')

plt.show()
# lets look at it in a logarithmic scale 

log_country_confirmed_cases = [math.log10(i) for i in country_confirmed_cases]

plt.figure(figsize=(32, 18))

plt.barh(unique_countries, log_country_confirmed_cases)

plt.title('Log of Common Log # of Coronavirus Confirmed Cases in Countries/Regions')

plt.show()
plt.figure(figsize=(32, 18))

plt.barh(unique_provinces, province_confirmed_cases)

plt.title('# of Coronavirus Confirmed Cases in Provinces/States')

plt.show()
log_province_confirmed_cases = [math.log10(i) for i in province_confirmed_cases]

plt.figure(figsize=(32, 18))

plt.barh(unique_provinces, log_province_confirmed_cases)

plt.title('Log of # of Coronavirus Confirmed Cases in Provinces/States')

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per Country')

plt.pie(country_confirmed_cases, colors=c)

plt.legend(unique_countries, loc='best')

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases per State/Province/Region')

plt.pie(province_confirmed_cases, colors=c)

plt.legend(unique_provinces, loc='best')

plt.show()
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in Countries Outside of Mainland China')

plt.pie(country_confirmed_cases[1:], colors=c)

plt.legend(unique_countries[1:], loc='best')

plt.show()
us_regions = list(confirmed_df[confirmed_df['Country/Region']=='US']['Province/State'].unique())

us_confirmed_cases = []

no_cases = [] 

for i in us_regions:

    cases = latest_confirmed[confirmed_df['Province/State']==i].sum()

    if cases > 0:

        us_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

# remove areas with no confirmed cases

for i in no_cases:

    us_regions.remove(i)
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in the United States')

plt.pie(us_confirmed_cases, colors=c)

plt.legend(us_regions, loc='best')

plt.show()
china_regions = list(confirmed_df[confirmed_df['Country/Region']=='Mainland China']['Province/State'].unique())

china_confirmed_cases = []

no_cases = [] 

for i in china_regions:

    cases = latest_confirmed[confirmed_df['Province/State']==i].sum()

    if cases > 0:

        china_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

 

# remove areas with no confirmed cases

for i in no_cases:

    china_confirmed_cases.remove(i)
c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))

plt.figure(figsize=(20,20))

plt.title('Covid-19 Confirmed Cases in the Mainland China')

plt.pie(china_confirmed_cases, colors=c)

plt.legend(china_regions, loc='best')

plt.show()