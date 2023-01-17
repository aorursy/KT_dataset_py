# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

#Aggregate by the store level for now

store_level = sales_train_validation.groupby(sales_train_validation['store_id']).sum()

store_level['d'] = store_level.index

store_levelt = store_level.transpose() 

store_levelt['d'] = store_levelt.index

store_levelt
#Merge this with the calendar data set to look at trends

store_level_final = store_levelt.merge(calendar, on='d')

from datetime import datetime

store_level_final['date'] = store_level_final['date'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d'))

store_level_final
plt.plot(store_level_final['date'][0:49], store_level_final['CA_1'][0:49])

plt.xticks(rotation=45)
plt.plot(store_level_final['weekday'][0:7], store_level_final['CA_1'][0:7])

plt.xticks(rotation=45)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):



    #Determing rolling statistics

    rolmean = timeseries.rolling(7).mean()

    rolstd = timeseries.rolling(7).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 8))

    orig = plt.plot(timeseries[0:100], color='blue',label='Original')

    mean = plt.plot(rolmean[0:100], color='red', label='Rolling Mean')

    std = plt.plot(rolstd[0:100], color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(store_level_final['CA_1'][0:100])


store_level_final['first_difference'] = store_level_final['CA_1'] - store_level_final['CA_1'].shift(1)

test_stationarity(store_level_final['first_difference'].dropna())
store_level_final['seasonal_difference'] = store_level_final['CA_1'] - store_level_final['CA_1'].shift(7)

test_stationarity(store_level_final['seasonal_difference'].dropna())
store_level_final['seasonal_first_difference'] = store_level_final['CA_1'] - store_level_final['CA_1'].shift(1) - store_level_final['CA_1'].shift(7) + store_level_final['CA_1'].shift(8)

test_stationarity(store_level_final['seasonal_first_difference'].dropna())

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(store_level_final['CA_1'][0:100])

autocorrelation_plot(store_level_final['seasonal_first_difference'][8:100])

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(store_level_final['seasonal_difference'][0:100].dropna(), lags=10)

plot_pacf(store_level_final['seasonal_difference'][0:100].dropna(), lags=10)

plot_acf(store_level_final['seasonal_first_difference'][0:100].dropna(), lags=10)

plot_pacf(store_level_final['seasonal_first_difference'][0:100].dropna(), lags=10)

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(store_level_final['CA_1'])
train = np.asarray(store_level_final['CA_1'][0:1880].astype(float))

test = np.asarray(store_level_final['CA_1'][1880:-1].astype(float))
import statsmodels.api as sm



mod = sm.tsa.statespace.SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,7))

results=mod.fit()

print(results.summary())
plt.plot(results.predict(start = 1800, end = 1880), label = 'Predicted')

plt.plot(train[1800:1880], label = 'Train')

plt.legend(loc = "upper left")
plt.plot(results.predict(start = len(train), end = len(store_level_final['CA_1'])), label = 'Predicted')

plt.plot(test, label = 'Test')

plt.legend(loc = 'upper left')
from sklearn.metrics import mean_squared_error

import math

mean_squared_error(results.predict(start = 1880, end = 1911), test)
train2 = np.asarray(store_level_final['CA_1'][1500:1880].astype(float))

test2 = np.asarray(store_level_final['CA_1'][1880:-1].astype(float))
import statsmodels.api as sm



mod2 = sm.tsa.statespace.SARIMAX(train2, order=(0,1,1), seasonal_order=(0,1,1,7))

results2=mod2.fit()

print(results2.summary())
plt.plot(results2.predict(start = 100, end = 200), label = 'Predicted')

plt.plot(train2[100:200], label = 'Train')

plt.legend(loc = "upper left")
plt.plot(results2.predict(start = 380, end = 411), label = 'Predicted')

plt.plot(test2, label = 'Test')

plt.legend(loc = "upper left")
mean_squared_error(results2.predict(start = 380, end = 411), test2)
store_level
category_sales = sales_train_validation.groupby(sales_train_validation['cat_id']).sum()

category_salest = category_sales.transpose()

category_salest['d'] = category_salest.index
#Merge this with the calendar data set to look at trends

category_level_final = category_salest.merge(calendar, on='d')

category_level_final['date'] = category_level_final['date'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d'))

category_level_final
category_level_model = category_level_final

category_level_model['FOODS'].loc[category_level_model['event_name_1'] == 'Christmas'] = category_level_final['FOODS'].mean()

category_level_model['HOBBIES'].loc[category_level_model['event_name_1'] == 'Christmas'] = category_level_final['HOBBIES'].mean()

category_level_model['HOUSEHOLD'].loc[category_level_model['event_name_1'] == 'Christmas'] = category_level_final['HOUSEHOLD'].mean()
train_food = np.asarray(category_level_model['FOODS'][0:1883].astype(float))

test_food = np.asarray(category_level_model['FOODS'][1883:-1].astype(float))

mod_food = sm.tsa.statespace.SARIMAX(train_food, order=(0,1,1), seasonal_order=(0,1,1,7))

results_food=mod_food.fit()

plt.plot(results_food.predict(start = len(train_food), end = 1911), label = 'Predicted')

plt.plot(test_food, label = 'Test')

plt.legend(loc = 'upper left')

rmse = math.sqrt(mean_squared_error(results_food.predict(start = 1883, end = 1911), test_food))

plt.title("Food: SARIMA Model Prediction vs. Test with RMSE = " + "{:.0f}".format(rmse))

rmse / np.std(train_food)
plt.plot(results_food.predict(start = 1800, end = 2200), label = 'Predicted')

train_hobbies = np.asarray(category_level_model['HOBBIES'][0:1883].astype(float))

test_hobbies = np.asarray(category_level_model['HOBBIES'][1883:-1].astype(float))

mod_hobbies = sm.tsa.statespace.SARIMAX(train_hobbies, order=(0,1,1), seasonal_order=(0,1,1,7))

results_hobbies = mod_hobbies.fit() 

plt.plot(results_hobbies.predict(start = len(train_hobbies), end = 1911), label = 'Predicted')

plt.plot(test_hobbies, label = 'Test')

plt.legend(loc = 'upper left')

rmse = math.sqrt(mean_squared_error(results_hobbies.predict(start = 1883, end = 1911), test_hobbies))

plt.title("Hobbies: SARIMA Model Prediction vs. Test with RMSE = " + "{:.0f}".format(rmse))
train_household = np.asarray(category_level_final['HOUSEHOLD'][0:1883].astype(float))

test_household = np.asarray(category_level_final['HOUSEHOLD'][1883:-1].astype(float))

mod_household = sm.tsa.statespace.SARIMAX(train_household, order=(0,1,1), seasonal_order=(0,1,1,7))

results_household = mod_household.fit()

plt.plot(results_household.predict(start = len(train_household), end = 1911), label = 'Predicted')

plt.plot(test_household, label = 'Test')

plt.legend(loc = 'upper left')

rmse = math.sqrt(mean_squared_error(results_household.predict(start = 1883, end = 1911), test_household))

plt.title("Household: SARIMA Model Prediction vs. Test with RMSE = " + "{:.0f}".format(rmse))
train_order = np.asarray(store_level_final['CA_1'][0:1000].astype(float))

test_order = np.asarray(store_level_final['CA_1'][1000:-1].astype(float))