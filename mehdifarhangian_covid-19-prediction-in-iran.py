import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

import xgboost

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 

plt.style.use('fivethirtyeight')

%matplotlib inline 
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-28-2020.csv')

cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

total_cases = []

total_deaths = [] 

mortality_rate = []

recovery_rate = [] 

total_recovered = [] 

total_active = [] 



 





for i in dates:

    confirmed_sum = confirmed_df[confirmed_df['Country/Region']=='Iran'][i].sum()

    death_sum = deaths_df[deaths_df['Country/Region']=='Iran'][i].sum()

    recovered_sum = recoveries_df[recoveries_df['Country/Region']=='Iran'][i].sum()

    

    # confirmed, deaths, recovered, and active

    total_cases.append(confirmed_sum)

    total_deaths.append(death_sum)

    total_recovered.append(recovered_sum)

    total_active.append(confirmed_sum-death_sum-recovered_sum)

    

    # calculate rates

    mortality_rate.append(death_sum/confirmed_sum)

    recovery_rate.append(recovered_sum/confirmed_sum)



    

  

    
def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



# confirmed cases





total_daily_increase = daily_increase(total_cases)



# deaths





total_daily_death = daily_increase(total_deaths)



# recoveries





total_daily_recovery = daily_increase(total_recovered)

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

total_cases = np.array(total_cases).reshape(-1, 1)

total_deaths = np.array(total_deaths).reshape(-1, 1)

total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 3

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-3]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, total_cases, test_size=0.30, shuffle=False) 
# svm_confirmed = svm_search.best_estimator_

svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=2, epsilon=1,degree=3, C=3)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(y_test_confirmed)

plt.plot(svm_test_pred)

plt.legend(['Test Data', 'SVM Predictions'])

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
adjusted_dates = adjusted_dates.reshape(1, -1)[0]

plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_deaths)

plt.title('# of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_recovered)

plt.title('# of Coronavirus Recoveries Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_active)

plt.title('# of Coronavirus Active Cases Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Active Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, total_daily_increase)

plt.title('Total Daily Increases in Confirmed Cases', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, total_daily_death)

plt.title('Total Daily Increases in Confirmed Deaths', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, total_daily_recovery)

plt.title('Total Daily Increases in Confirmed Recoveries', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(total_cases))

plt.title('Log of # of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(total_deaths))

plt.title('Log of # of Coronavirus Deaths Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()



plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(total_recovered))

plt.title('Log of # of Coronavirus Recoveries Over Time', size=30)

plt.xlabel('Days Since 22 Jan of 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
def country_plot(x, y1, y2, y3, y4, country):

    plt.figure(figsize=(16, 9))

    plt.plot(x, y1)

    plt.title('{} Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 22 Jan of 2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()



    plt.figure(figsize=(16, 9))

    plt.bar(x, y2)

    plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 22 Jan of 2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()



    plt.figure(figsize=(16, 9))

    plt.bar(x, y3)

    plt.title('{} Daily Increases in Deaths'.format(country), size=30)

    plt.xlabel('Days Since 22 Jan of 2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()



    plt.figure(figsize=(16, 9))

    plt.bar(x, y4)

    plt.title('{} Daily Increases in Recoveries'.format(country), size=30)

    plt.xlabel('Days Since 22 Jan of 2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
country_plot(adjusted_dates, total_cases, total_daily_increase, total_daily_death, total_daily_recovery, 'Iran')
def plot_predictions(x, y, pred, algo_name, color):

    plt.figure(figsize=(16, 9))

    plt.plot(x, y)

    plt.plot(future_forcast, pred, linestyle='dashed', color=color)

    plt.title('# of Coronavirus Cases Over Time', size=30)

    plt.xlabel('Days Since 22 Jan 2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
plot_predictions(adjusted_dates, total_cases, svm_pred, 'SVM Predictions', 'purple')
# Future predictions using SVM 

svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'SVM Predicted # of Confirmed Cases': np.round(svm_pred[-10:])})

svm_df
mean_recovery_rate = np.mean(recovery_rate)

plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, recovery_rate, color='blue')

plt.axhline(y = mean_recovery_rate,linestyle='--', color='black')

plt.title('Recovery Rate of Coronavirus Over Time', size=30)

plt.legend(['recovery rate', 'y='+str(mean_recovery_rate)], prop={'size': 20})

plt.xlabel('Days Since  22 Jan 2020', size=30)

plt.ylabel('Recovery Rate', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['death', 'recoveries'], loc='best', fontsize=20)

plt.title('# of Coronavirus Cases', size=30)

plt.xlabel('Days Since 22 Jan 2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(total_recovered, total_deaths)

plt.title('# of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)

plt.xlabel('# of Coronavirus Recoveries', size=30)

plt.ylabel('# of Coronavirus Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()