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
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

plt.style.use('seaborn')

%matplotlib inline 
%%capture

from tqdm import tqdm_notebook as tqdm

tqdm().pandas()
recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

death = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

covid_OpenList = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

covid_Linelist = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

covid_Data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
recovered.sample(2)
cols = recovered.keys()
confirmed.sample(2)
confirmed_df = confirmed.loc[:,cols[4]:cols[-1]]
recovered_df = recovered.loc[:,cols[4]:cols[-1]]

death_df = death.loc[:,cols[4]:cols[-1]]
dates = confirmed_df.keys()
total_recovered=[]

total_confirmed=[]

total_dead=[]

total_diseased=[]

mortality_rate=[]
for i in dates:

    date_recovered = recovered_df[i].sum()

    date_confirmed = confirmed_df[i].sum()

    date_dead = death_df[i].sum()

    date_diseased = date_confirmed - date_recovered - date_dead

    total_recovered.append(date_recovered)

    total_confirmed.append(date_confirmed)

    total_dead.append(date_dead)

    mortality_rate.append(date_dead/date_confirmed)

    total_diseased.append(date_diseased)
days_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

confirmed_cases = np.array(total_confirmed).reshape(-1, 1)

death_cases = np.array(total_dead).reshape(-1, 1)

recovered_cases = np.array(total_recovered).reshape(-1, 1)

mortality_rate_cases = np.array(mortality_rate).reshape(-1, 1)

still_diseased_cases = np.array(total_diseased).reshape(-1, 1)
days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-5]
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_1_22, confirmed_cases, test_size=0.15) 
svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.1, epsilon=1, C=0.01)

svm_confirmed.fit(X_train_confirmed, y_train_confirmed)

svm_pred = svm_confirmed.predict(future_forcast)
svm_pred
mean_squared_error(y_test_confirmed, svm_pred)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))

print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
gbr = GradientBoostingRegressor()
# gbr.fit(X_train_confirmed, y_train_confirmed)
# mean_squared_error(y_test_confirmed, gbr.predict(X_test_confirmed))
# # check against testing data

# gbr_test_pred = gbr.predict(X_test_confirmed)

# plt.plot(gbr_test_pred)

# plt.plot(y_test_confirmed)

# print('MAE:', mean_absolute_error(gbr_test_pred, y_test_confirmed))

# print('MSE:',mean_squared_error(gbr_test_pred, y_test_confirmed))
rfr = RandomForestRegressor(n_estimators=200)
# rfr.fit(X_train_confirmed, y_train_confirmed)
# # check against testing data

# rfr_test_pred = rfr.predict(X_test_confirmed)

# plt.plot(rfr_test_pred)

# plt.plot(y_test_confirmed)

# print('MAE:', mean_absolute_error(rfr_test_pred, y_test_confirmed))

# print('MSE:',mean_squared_error(rfr_test_pred, y_test_confirmed))
def createNewDates(date_input, no_ofdays):

    days_future = no_ofdays

    future_forcast = np.array([i for i in range(days_future)]).reshape(-1, 1)

    end = date_input

    end_date = datetime.datetime.strptime(end, '%m/%d/%Y')

    forcast_dates = []

    for i in range(1,len(future_forcast)+1):

        forcast_dates.append((end_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

    return forcast_dates
def model_(modelname, x, y):

    mod = modelname()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

    mod.fit(xtrain,ytrain)

    return mod
x = np.array([i for i in range(len(days_1_22)+2)]).reshape(-1, 1)
svm_confirmed_mod = model_(SVR, days_1_22, confirmed_cases)

svm_recovered_mod = model_(SVR, days_1_22, recovered_cases)

svm_dead_mod = model_(SVR, days_1_22, death_cases)

svm_mortality_mod = model_(SVR, days_1_22, mortality_rate_cases)

svm_diseased_mod = model_(SVR, days_1_22, still_diseased_cases)

# randomForest_recovered = model_(rfr, days_1_22, still_diseased_cases)
randomForest_recovered.feature_importances_
svm_confirmed_mod.predict(x)
randomdate = '3/14/2020'

day_diff = datetime.datetime.strptime(future_df.iloc[0, 0], '%m/%d/%Y') - datetime.datetime.strptime(randomdate, '%m/%d/%Y')
future_df = pd.DataFrame({'Dates':createNewDates('3/14/2020', 10)})
future_df.iloc[0, 0]
future_df['Confirmed'] = None

future_df['Recovered'] = None

future_df['Dead'] = None

future_df['Mortality Rate'] = None

future_df['Diseased'] = None
for index, row in tqdm(future_df.iterrows()):

    index_date = row['Dates']

    if type(index_date)!=float:

        day_diff = (datetime.datetime.strptime(index_date, '%m/%d/%Y') - datetime.datetime.strptime(randomdate, '%m/%d/%Y')).days

        countDays = day_diff+days_1_22

        print(index_date)

        future_df.loc[index, 'Confirmed'] = svm_confirmed.predict(countDays)[-1]

        future_df.loc[index, 'Recovered'] = svm_recovered_mod.predict(countDays)[-1]

        future_df.loc[index, 'Dead'] = svm_dead_mod.predict(countDays)[-1]

        future_df.loc[index, 'Mortality Rate'] = svm_mortality_mod.predict(countDays)[-1]

        future_df.loc[index, 'Diseased'] = svm_diseased_mod.predict(countDays)[-1]
future_df.to_csv('/kaggle/working/predictionfornext10days_24032020.csv', index=None)