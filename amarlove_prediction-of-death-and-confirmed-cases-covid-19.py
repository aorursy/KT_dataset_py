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

confirmed_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recoveries_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
confirmed_df.head()
columns = confirmed_df.keys()
confirmed = confirmed_df.loc[:, columns[4]:columns[-1]]

deaths = deaths_df.loc[:, columns[4]:columns[-1]]

recoveries = recoveries_df.loc[:, columns[4]:columns[-1]]
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
days_in_future = 15

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-15]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
kernel = ['poly', 'sigmoid', 'rbf']

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
# checking against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
linear_model = LinearRegression(normalize=True, fit_intercept=True)

linear_model.fit(X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(X_test_confirmed)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)

print(linear_model.intercept_)
tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(X_test_confirmed)

bayesian_pred = bayesian_confirmed.predict(future_forcast)

print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)

plt.plot(test_bayesian_pred)
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(20, 12))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Time', size=30)

plt.ylabel('# of Cases', size=30)

plt.legend(['Confirmed Cases', 'Bayesian Ridge Regression Predictions'])

plt.xticks(size=15)

plt.show()
# Future predictions using SVM 

print('SVM future predictions:')

set(zip(future_forcast_dates[-15:], svm_pred[-15:]))
# Future predictions using Linear Regression 

print('Ridge regression future predictions:')

set(zip(future_forcast_dates[-15:], bayesian_pred[-15:]))
# Future predictions using Linear Regression 

print('Linear regression future predictions:')

print(linear_pred[-15:])
# Split data for model

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths = train_test_split(days_since_1_22, total_deaths, test_size=0.15, shuffle=False) 
kernel = ['poly', 'sigmoid', 'rbf']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, return_train_score=True, n_iter=40, verbose=1)

svm_search.fit(X_train_deaths, y_train_deaths)
print('Best Params are: ')

svm_search.best_params_
svm_deaths = svm_search.best_estimator_

svm_pred_death = svm_deaths.predict(future_forcast)
# check against testing data

svm_test_pred = svm_deaths.predict(X_test_deaths)

plt.plot(svm_test_pred)

plt.plot(y_test_deaths)

plt.legend(['Death Cases', 'SVM predictions'])

print('MAE:', mean_absolute_error(svm_test_pred, y_test_deaths))

print('MSE:',mean_squared_error(svm_test_pred, y_test_deaths))
linear_model = LinearRegression(normalize=True, fit_intercept=True)

linear_model.fit(X_train_deaths, y_train_deaths)

test_linear_pred = linear_model.predict(X_test_deaths)

linear_pred = linear_model.predict(future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_deaths))

print('MSE:',mean_squared_error(test_linear_pred, y_test_deaths))
plt.plot(y_test_deaths)

plt.plot(test_linear_pred)

plt.legend(['Death Cases', 'Linear Regression predictions'])
tol = [1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}



bayesian = BayesianRidge()

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(X_train_deaths, y_train_deaths)
bayesian_search.best_params_
bayesian_deaths = bayesian_search.best_estimator_

test_bayesian_pred_deaths = bayesian_deaths.predict(X_test_deaths)

bayesian_pred_deaths = bayesian_deaths.predict(future_forcast)

print('MAE:', mean_absolute_error(test_bayesian_pred_deaths, y_test_deaths))

print('MSE:',mean_squared_error(test_bayesian_pred_deaths, y_test_deaths))
plt.plot(y_test_deaths)

plt.plot(test_bayesian_pred_deaths)

plt.legend(['Confirmed Cases', 'Bayesian predictions'])
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.title('# of Coronavirus Death Cases Over Time', size=20)

plt.xlabel('Time', size=20)

plt.ylabel('# of Deaths', size=20)

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.plot(future_forcast, svm_pred_death, linestyle='dashed', color='purple')

plt.title('# of Coronavirus Death Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Death Cases', 'SVM predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Death Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Death Cases', 'Linear Regression Predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, total_deaths, color='red')

plt.plot(future_forcast, bayesian_pred_deaths, linestyle='dashed', color='green')

plt.title('# of Coronavirus Death Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Death Cases', 'Bayesian Ridge Regression Predictions'])

plt.xticks(size=15)

plt.show()
# Future predictions using SVM 

print('SVM future predictions:')

set(zip(future_forcast_dates[-14:], svm_pred_death[-14:]))
# Future predictions using Bayesian regression

print('Bayesian regression future predictions:')

set(zip(future_forcast_dates[-14:], bayesian_pred_deaths[-14:]))
# Future predictions using Linear Regression 

print('Linear regression future predictions:')

print(linear_pred[-14:])
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, total_deaths, color='r')

plt.plot(adjusted_dates, total_recovered, color='green')

plt.legend(['Deaths', 'Recoveries'], loc='best', fontsize=20)

plt.title('# of Coronavirus Cases', size=20)

plt.xlabel('Time', size=20)

plt.ylabel('# of Cases', size=20)

plt.xticks(size=15)

plt.show()