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

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 
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
data_confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

data_confirmed.head()
cols = data_confirmed.keys()
confirmed = data_confirmed.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()

world_cases = []



us_cases = [] 
for i in dates:

    

    confirmed_sum = confirmed[i].sum()

    

    world_cases.append(confirmed_sum)

    

    us_cases.append(data_confirmed[data_confirmed['Country/Region']=='US'][i].sum())

    
# Daily Increase

def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 

# Confirmed cases:



world_daily_increase = daily_increase(world_cases)



us_daily_increase = daily_increase(us_cases)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)



days_since_1_22.shape



world_cases.shape
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.07, shuffle=False)
# SVM



svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=4, C=0.1)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(y_test_confirmed)

plt.plot(svm_test_pred)

plt.legend(['Test Data', 'SVM Predictions'])

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# Using polynomial Regression to tranforom the data



poly = PolynomialFeatures(degree=3)

poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

poly_future_forcast = poly.fit_transform(future_forcast)



bayesian_poly = PolynomialFeatures(degree=4)

bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)

bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)

bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
# polynomial regression

linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(poly_X_test_confirmed)

linear_pred = linear_model.predict(poly_future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
plt.plot(y_test_confirmed)

plt.plot(test_linear_pred)

plt.legend(['Test Data', 'Polynomial Regression Predictions'])
# Bayesian ridge polynomial regression 



tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

normalize = [True, False]



bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 

                 'normalize' : normalize}



bayesian = BayesianRidge(fit_intercept=False)

bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
bayesian_confirmed = bayesian_search.best_estimator_

test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)

bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)

print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)

plt.plot(test_bayesian_pred)

plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
adjusted_dates = adjusted_dates.reshape(1, -1)[0]

plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, world_cases)

plt.title('# of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()

plt.figure(figsize=(16, 9))

plt.bar(adjusted_dates, world_daily_increase)

plt.title('World Daily Increases in Confirmed Cases', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
plt.figure(figsize=(16, 9))

plt.plot(adjusted_dates, np.log10(world_cases))

plt.title('Log of # of Coronavirus Cases Over Time', size=30)

plt.xlabel('Days Since 1/22/2020', size=30)

plt.ylabel('# of Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()

def country_plot(x, y1, y2,country):

    plt.figure(figsize=(16, 9))

    plt.plot(x, y1)

    plt.title('{} Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()

    

    plt.figure(figsize=(16, 9))

    plt.bar(x, y2)

    plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
country_plot(adjusted_dates, us_cases, us_daily_increase, 'United States')
def plot_predictions(x, y, pred, algo_name, color):

    plt.figure(figsize=(16, 9))

    plt.plot(x, y)

    plt.plot(future_forcast, pred, linestyle='dashed', color=color)

    plt.title('# of Coronavirus Cases Over Time', size=30)

    plt.xlabel('Days Since 1/22/2020', size=30)

    plt.ylabel('# of Cases', size=30)

    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()
plot_predictions(adjusted_dates, world_cases, svm_pred, 'SVM Predictions', 'purple')
plot_predictions(adjusted_dates, world_cases, linear_pred, 'Polynomial Regression Predictions', 'orange')
plot_predictions(adjusted_dates, world_cases, bayesian_pred, 'Bayesian Ridge Regression Predictions', 'green')
# Future predictions using SVM 

svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'SVM Predicted # of Confirmed Cases Worldwide': np.round(svm_pred[-10:])})

svm_df
# Future predictions using polynomial regression

linear_pred = linear_pred.reshape(1,-1)[0]

svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted # of Confirmed Cases Worldwide': np.round(linear_pred[-10:])})

svm_df
# Future predictions using Bayesian Ridge 

svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted # of Confirmed Cases Worldwide': np.round(bayesian_pred[-10:])})

svm_df