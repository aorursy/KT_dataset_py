# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

import seaborn as sns

import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error

%matplotlib inline
# Loading the data

deaths_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_deaths_global.csv')

deaths_global.head()
col= deaths_global.columns

col
deaths_global = deaths_global.loc[:, col[4]:col[-1]]

deaths_global.head()
#total number of deaths on 1/22/2020 in the world.

deaths_global['1/22/20'].sum()
#total number of deaths on 7/13/2020 in the world.

deaths_global['7/13/20'].sum()
#creating the list of total number of deaths in the world.

dates = deaths_global.keys()

y_deaths = []

for i in dates:

    y_deaths.append(deaths_global[i].sum())
print(y_deaths)
#will be transformed into matrix format

y_deaths = np.array(y_deaths).reshape(-1,1) 

print(y_deaths)
#will generate the number of dates according to the size of the "dates"

x_deaths = np.arange(len(dates)).reshape(-1,1) 

print(x_deaths)
#will generate the number of dates according to the size of the "dates"+20 to forecast the number cases for those days

forecast_deaths = np.arange(len(dates) + 20).reshape(-1,1)

print(forecast_deaths)
print('shape of x:       ',x_deaths.shape)

print('shape of y:       ',y_deaths.shape)

print('shape of forecast:',forecast_deaths.shape)
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forecast_dates = []

for i in range(len(forecast_deaths)):

    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
from sklearn.model_selection import train_test_split

X_train_deaths, X_test_deaths, Y_train_deaths, Y_test_deaths = train_test_split(x_deaths, y_deaths, test_size = 0.15, shuffle = False)
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree = 2)

X_train_deaths_pf = pf.fit_transform(X_train_deaths)

X_test_deaths_pf = pf.transform(X_test_deaths)
print('Shape of X_train:   ',X_train_deaths.shape)

print('Shape of X_train_pf:', X_train_deaths_pf.shape)

print('Shape of X_test:    ', X_test_deaths.shape)

print('Shape of X_test_pf: ', X_test_deaths_pf.shape)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_deaths_pf, Y_train_deaths)
pf_pred_deaths = lr.predict(X_test_deaths_pf)

plt.plot(pf_pred_deaths, linestyle = 'dashed')

plt.plot(Y_test_deaths);

plt.legend(['Predicted','y_test']);
print('MAE:', mean_absolute_error(pf_pred_deaths, Y_test_deaths))

print('MSE:', mean_squared_error(pf_pred_deaths, Y_test_deaths))

print('RMSE:', np.sqrt(mean_absolute_error(pf_pred_deaths, Y_test_deaths)))
X_train_deaths_all = pf.transform(forecast_deaths)

pred_deaths_all = lr.predict(X_train_deaths_all)



plt.figure(figsize=(8,6))

plt.plot(forecast_deaths[:-20], y_deaths, color='red')

plt.plot(forecast_deaths, pred_deaths_all, linestyle='dashed')

plt.title('DEATHS of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of deaths')

plt.legend(['Death cases', 'Predictions']);
#creating dataframe for the Predicted no. of Deaths for 20 days.

pred_deaths_all = pred_deaths_all.reshape(1,-1)[0]

predicted_death_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'Predicted no. of Deaths': np.round(pred_deaths_all[-20:])})

predicted_death_df
#Loading the data

confirmed_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_confirmed_global.csv')

confirmed_global.head()
column = confirmed_global.columns

column
confirmed_global = confirmed_global.loc[:, column[4]:column[-1]]

confirmed_global.head()
# Total number of confirmed cases on the given date in the world

confirmed_global['1/22/20'].sum(),confirmed_global['7/13/20'].sum()
#creating the list of total number of deaths in the world.

dates_c = confirmed_global.keys()

y_confirmed = []

for i in dates:

    y_confirmed.append(confirmed_global[i].sum())
print(y_confirmed)
#will be transformed into matrix format

y_confirmed = np.array(y_confirmed).reshape(-1,1) 

print(y_confirmed)
#will generate the number of dates according to the size of the "dates"

x_confirmed = np.arange(len(dates_c)).reshape(-1,1) 

print(x_confirmed)
#will generate the number of dates according to the size of the "dates"+20 to forecast the number cases for those days

forecast_confirmed = np.arange(len(dates_c) + 20).reshape(-1,1)

print(forecast_confirmed)
print('shape of x:       ',x_confirmed.shape)

print('shape of y:       ',y_confirmed.shape)

print('shape of forecast:',forecast_confirmed.shape)
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forecast_dates_c = []

for i in range(len(forecast_confirmed)):

    future_forecast_dates_c.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, Y_train_confirmed, Y_test_confirmed = train_test_split(x_confirmed, y_confirmed, test_size = 0.15, shuffle = False)
pf_c = PolynomialFeatures(degree = 2)

X_train_confirmed_pf = pf_c.fit_transform(X_train_confirmed)

X_test_confirmed_pf = pf_c.transform(X_test_confirmed)
print('Shape of X_train:   ',X_train_confirmed.shape)

print('Shape of X_train_pf:', X_train_confirmed_pf.shape)

print('Shape of X_test:    ', X_test_confirmed.shape)

print('Shape of X_test_pf: ', X_test_confirmed_pf.shape)
lr_c = LinearRegression()

lr_c.fit(X_train_confirmed_pf, Y_train_confirmed)
pf_pred_confirmed = lr_c.predict(X_test_confirmed_pf)

plt.plot(pf_pred_confirmed, linestyle = 'dashed')

plt.plot(Y_test_confirmed);

plt.legend(['Predicted','Y_test'])
print('MAE:', mean_absolute_error(pf_pred_confirmed, Y_test_confirmed))

print('MSE:', mean_squared_error(pf_pred_confirmed, Y_test_confirmed))

print('RMSE:', np.sqrt(mean_absolute_error(pf_pred_confirmed, Y_test_confirmed)))
X_train_confirmed_all = pf_c.transform(forecast_confirmed)

pred_confirmed_all = lr_c.predict(X_train_confirmed_all)



plt.figure(figsize=(8,6))

plt.plot(forecast_confirmed[:-20], y_confirmed, color='brown')

plt.plot(forecast_confirmed, pred_confirmed_all, linestyle='dashed')

plt.title('CONFIRMED Cases of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of Confirmed Cases')

plt.legend(['Confirmed cases', 'Predictions']);
#creating dataframe for the Predicted no. of Confirmed cases for next 20 days.

pred_confirmed_all = pred_confirmed_all.reshape(1,-1)[0]

predicted_confirmed_df = pd.DataFrame({'Date': future_forecast_dates_c[-20:], 'Predicted no. of Confirmed Cases': np.round(pred_confirmed_all[-20:])})

predicted_confirmed_df
#Loading the data

recovered_global= pd.read_csv('/kaggle/input/covid19-assignment/time_series_covid19_recovered_global.csv')

recovered_global.head()
columns2 = recovered_global.columns

columns2
recovered_global = recovered_global.loc[:, columns2[4]:columns2[-1]]

recovered_global.head()
# Total number of recovered cases on the given date in the world.

recovered_global['1/22/20'].sum(), recovered_global['7/13/20'].sum()
#creating the list of total number of recovered cases in the world.

dates2 = recovered_global.keys()

y_recovered = []

for i in dates2:

    y_recovered.append(recovered_global[i].sum())
print(y_recovered)
#will be transformed into matrix format

y_recovered = np.array(y_recovered).reshape(-1,1)

y_recovered
#will generate the number of dates according to the size of the "dates"

x_recovered = np.arange(len(dates2)).reshape(-1,1)

x_recovered
#will generate the number of dates according to the size of the "dates"+20 to forecast the number cases for those days

forecast_recovered = np.arange(len(dates2) + 20).reshape(-1,1)

forecast_recovered
print('shape of x:       ',x_recovered.shape)

print('shape of y:       ',y_recovered.shape)

print('shape of forecast:',forecast_recovered.shape)
start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forecast_dates_r = []

for i in range(len(forecast_recovered)):

    future_forecast_dates_r.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_recovered, X_test_recovered, Y_train_recovered, Y_test_recovered = train_test_split(x_recovered, y_recovered, test_size = 0.15, shuffle = False)
pf_r = PolynomialFeatures(degree = 3)

X_train_pf_recovered = pf_r.fit_transform(X_train_recovered)

X_test_pf_recovered = pf_r.transform(X_test_recovered)
print('Shape of X_train:   ',X_train_recovered.shape)

print('Shape of X_train_pf:', X_train_pf_recovered.shape)

print('Shape of X_test:    ', X_test_recovered.shape)

print('Shape of X_test_pf: ', X_test_pf_recovered.shape)
lr_r = LinearRegression()

lr_r.fit(X_train_pf_recovered, Y_train_recovered)
pf_pred_recovered = lr_r.predict(X_test_pf_recovered)

plt.plot(pf_pred_recovered, linestyle = 'dashed')

plt.plot(Y_test_recovered)

plt.legend(['Predicted','Y_test']);
print('MAE:', mean_absolute_error(pf_pred_recovered, Y_test_recovered))

print('MSE:', mean_squared_error(pf_pred_recovered, Y_test_recovered))

print('RMSE:', np.sqrt(mean_absolute_error(pf_pred_recovered, Y_test_recovered)))
X_train_all_recovered = pf_r.transform(forecast_recovered)

pred_all_recovered = lr_r.predict(X_train_all_recovered)



plt.figure(figsize=(8,6))

plt.plot(forecast_recovered[:-20], y_recovered, color='red')

plt.plot(forecast_recovered, pred_all_recovered, linestyle='dashed')

plt.title('RECOVERED Cases of COVID-19')

plt.xlabel('Days since 1/22/2020')

plt.ylabel('Number of recovered')

plt.legend(['Recovered', 'Predictions']);
#creating dataframe for the Predicted no. of Confirmed cases for next 20 days.

pred_all_recovered = pred_all_recovered.reshape(1,-1)[0]

predicted_recovered_df = pd.DataFrame({'Date': future_forecast_dates_r[-20:], 'Predicted no. of Recovered Cases': np.round(pred_all_recovered[-20:])})

predicted_recovered_df
temp= predicted_confirmed_df.merge(predicted_death_df,on='Date',how='right')



final_df= temp.merge(predicted_recovered_df,on='Date',how='right')

final_df