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
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
confirmed_df.head()
confirmed_df.describe()
confirmed_df
cols = confirmed_df.keys()
cols
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]

deaths = deaths_df.loc[:, cols[4]:cols[-1]]

recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
total = [confirmed_df, deaths_df, recoveries_df]

result = pd.concat([confirmed_df, deaths_df, recoveries_df], axis=1, join='inner')

result

total_one = [confirmed_df, deaths_df, recoveries_df]

result = pd.concat([confirmed_df, deaths_df, recoveries_df], axis=1, join='outer')

result
result.keys()
from collections import defaultdict



dd = defaultdict(list)



for d in (confirmed_df, deaths_df, recoveries_df):

    for key, value in d.items():

        dd[key].append(value)
dd
confirmed


dates = confirmed.keys()

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
world_cases
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
future_forcast_dates
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# transform our data for polynomial regression

poly = PolynomialFeatures(degree=5)

poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)

poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)

poly_future_forcast = poly.fit_transform(future_forcast)
linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train_confirmed, y_train_confirmed)

test_linear_pred = linear_model.predict(poly_X_test_confirmed)

linear_pred = linear_model.predict(poly_future_forcast)

print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))

print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)
plt.plot(test_linear_pred)

plt.plot(y_test_confirmed)
X_features = ['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',

       '3/15/20', '3/16/20', '3/17/20', '3/18/20', '3/19/20', '3/20/20',

       '3/21/20', '3/22/20']

X = confirmed_df[X_features]

X.head()

X = X.fillna(method = 'ffill')

X.isnull()
Y = confirmed_df['3/23/20']

Y.head()

Y = Y.fillna(method = 'ffill')
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

covidTree_model = DecisionTreeRegressor(random_state=1)



# Fit model

covidTree_model.fit(X, Y)


print(X.head())

print("The predictions are")

print(covidTree_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error



predicted_cases = covidTree_model.predict(X)

mean_absolute_error(Y, predicted_cases)
from sklearn.model_selection import train_test_split



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)

# Define model

covidTree_model = DecisionTreeRegressor()

# Fit model

covidTree_model.fit(train_X, train_y)



# get predicted prices on validation data

val_predictions = covidTree_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))