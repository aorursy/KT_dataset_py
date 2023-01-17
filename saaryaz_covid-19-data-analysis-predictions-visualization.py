# install calmap

! pip install calmap



# essential libraries

import json

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import calmap

import folium



# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px



from sklearn.cluster import KMeans

from fbprophet.plot import plot_plotly, add_changepoints_to_plot

import plotly.offline as py

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

from keras.models import Sequential

from keras.layers import LSTM,Dense

from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator



# Setup

import folium

import plotly.graph_objects as go

import math 

import random



# for time series

from fbprophet import Prophet

from fbprophet.plot import plot_plotly









########

import matplotlib.colors as mcolors

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator

%matplotlib inline 

full_table = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

full_table.head()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)
# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Province/State'].str.contains('Diamond Princess cruise ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()

temp.style.background_gradient(cmap='Reds')
temp_flg = temp_f[temp_f['Deaths']>0][['Country/Region', 'Deaths']]

temp_flg.sort_values('Deaths', ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
# World wide



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)



for i in range(0, len(full_latest)):

    folium.Circle(

        location=[full_latest.iloc[i]['Lat'], full_latest.iloc[i]['Long']],

        color='crimson', 

        tooltip =   '<li><bold>Country : '+str(full_latest.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(full_latest.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(full_latest.iloc[i]['Confirmed'])+

                    '<li><bold>Deaths : '+str(full_latest.iloc[i]['Deaths'])+

                    '<li><bold>Recovered : '+str(full_latest.iloc[i]['Recovered']),

        radius=int(full_latest.iloc[i]['Confirmed'])**1.1).add_to(m)

m
patient_path_13_3_20 = '/kaggle/input/coronavirusdataset/patient.csv'

time_path_13_3_20 = '/kaggle/input/coronavirusdataset/time.csv'

route_path_13_3_20 = '/kaggle/input/coronavirusdataset/route.csv'
patient_13_3_20 = pd.read_csv(patient_path_13_3_20, index_col="patient_id")

time_13_3_20 = pd.read_csv(time_path_13_3_20, index_col="date")

route_13_3_20 = pd.read_csv(route_path_13_3_20, index_col="patient_id")



print(f"Last Update: {pd.datetime.today().strftime('%m/%d/%Y')}")
patient_13_3_20.info()
patient_13_3_20.isna().sum()
dead_13_3_20 = patient_13_3_20[patient_13_3_20.state == 'deceased']

dead_13_3_20.head()
patient_13_3_20['birth_year'] = patient_13_3_20.birth_year.fillna(0.0).astype(int)

patient_13_3_20['birth_year'] = patient_13_3_20['birth_year'].map(lambda val: val if val > 0 else np.nan)



dead_13_3_20['age'] = 2020 - patient_13_3_20['birth_year'] 



plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the deceased")

ax = sns.kdeplot(data=dead_13_3_20['age'], shade=True, color="g")

male_dead_13_3_20 = dead_13_3_20[dead_13_3_20.sex == 'male']

female_dead_13_3_20 = dead_13_3_20[dead_13_3_20.sex == 'female']



plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the deceased by gender")

sns.kdeplot(data=female_dead_13_3_20['age'], label="Female", shade=True).set(xlim=(0))

sns.kdeplot(data=male_dead_13_3_20['age'],label="Male" ,shade=True).set(xlim=(0))
female_data=female_dead_13_3_20['age']

male_data=male_dead_13_3_20['age']



plt.hist(female_data , bins=10, label="Female")

plt.hist(male_data, bins=10, label="Male")



plt.legend()



plt.title("Age distribution of the deceased by gender")

plt.xlabel("Age")

plt.show()

female_data=female_dead_13_3_20['age']

male_data=male_dead_13_3_20['age']



plt.hist(female_data , bins=20)

plt.hist(male_data, bins=20)



plt.title("Age distribution of the deceased by gender")

plt.xlabel("Age")

plt.show()

released_13_3_20 = patient_13_3_20[patient_13_3_20.state == 'released']

released_13_3_20['age'] = 2020 - patient_13_3_20['birth_year'] 



male_released_13_3_20 = released_13_3_20[released_13_3_20.sex == 'male']

female_released_13_3_20 = released_13_3_20[released_13_3_20.sex == 'female']



plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the released by gender")

sns.kdeplot(data=female_released_13_3_20['age'], label="Female", shade=True).set(xlim=(0))

sns.kdeplot(data=male_released_13_3_20['age'],label="Male" ,shade=True).set(xlim=(0))
female_data=female_released_13_3_20['age']

male_data=male_released_13_3_20['age']



plt.hist(female_data , bins=10)

plt.hist(male_data, bins=10)



plt.title("Age distribution of the deceased by gender")

plt.xlabel("Age")

plt.show()

female_data=female_released_13_3_20['age']

male_data=male_released_13_3_20['age']



plt.hist(female_data , bins=20)

plt.hist(male_data, bins=20)



plt.title("Age distribution of the deceased by gender")

plt.xlabel("Age")

plt.show()
import matplotlib.colors as mcolors

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

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

print('Days Since 22.1.20: ' , len(days_since_1_22))
# Forcasting values

days_in_future = 14

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-14]
# Convert integer into datetime for better visualization

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
# Split data for model

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False) 
kernel = ['poly', 'sigmoid', 'rbf']

c = [0.01, 0.1, 1, 10]

gamma = [0.01, 0.1, 1]

epsilon = [0.01, 0.1, 1]

shrinking = [True, False]

svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}



svm = SVR()

svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=100, verbose=1)

svm_search.fit(X_train_confirmed, y_train_confirmed)
print('Best Params are: ')

svm_search.best_params_
svm_confirmed = svm_search.best_estimator_

svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data

svm_test_pred = svm_confirmed.predict(X_test_confirmed)

plt.plot(svm_test_pred)

plt.plot(y_test_confirmed)

plt.legend(['Confirmed Cases', 'SVM predictions'])

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
plt.plot(y_test_confirmed)

plt.plot(test_linear_pred)

plt.legend(['Confirmed Cases', 'Linear Regression predictions'])
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

plt.legend(['Confirmed Cases', 'Bayesian predictions'])
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, world_cases)

plt.title('# of Coronavirus Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, svm_pred, linestyle='dashed', color='purple')

plt.title('# of Coronavirus Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Confirmed Cases', 'SVM predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, linear_pred, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Confirmed Cases', 'Linear Regression Predictions'])

plt.xticks(size=15)

plt.show()
plt.figure(figsize=(10, 7))

plt.plot(adjusted_dates, world_cases)

plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')

plt.title('# of Coronavirus Cases Over Time', size=20)

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('# of Cases', size=20)

plt.legend(['Confirmed Cases', 'Bayesian Ridge Regression Predictions'])

plt.xticks(size=15)

plt.show()
# Future predictions using SVM 

print('SVM future predictions:')

set(zip(future_forcast_dates[-14:], svm_pred[-14:]))
# Future predictions using Bayesian regression

print('Bayesian regression future predictions:')

set(zip(future_forcast_dates[-14:], bayesian_pred[-14:]))
# Future predictions using Linear Regression 

print('Linear regression future predictions:')

print(linear_pred[-14:])
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
print(linear_model.coef_)

print(linear_model.intercept_)
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