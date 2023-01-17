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
import plotly.express as px
import matplotlib.dates as mdates
plt.style.use('fivethirtyeight')
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-09-2020.csv')
us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/08-09-2020.csv')
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow
latest_data.head()

confirmed_df.head()
us_medical_data.head()
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 
 
us_cases = []  
india_cases = []
# 
 
us_deaths = []  
india_deaths = [] 

us_recoveries = []  
india_recoveries = [] 
 

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
    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum()) 
    india_cases.append(confirmed_df[confirmed_df['Country/Region']=='India'][i].sum())
    
    # moving average for case studies  
    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum()) 
    india_deaths.append(deaths_df[deaths_df['Country/Region']=='India'][i].sum()) 
    
    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum()) 
    india_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='India'][i].sum())
days = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
# slightly modify the data to fit the model better (regression models cannot pick the pattern)
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days[50:], world_cases[50:], test_size=0.12, shuffle=False) 
# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=5)
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
print(linear_model.coef_)

plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
# bayesian ridge polynomial regression
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
bayesian_search.best_params_

bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
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
unique_countries =  list(latest_data['Country_Region'].unique())
country_confirmed_cases = []
country_death_cases = [] 
country_active_cases = []
country_recovery_cases = []
country_mortality_rate = [] 

no_cases = []
for i in unique_countries:
    cases = latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
# sort countries by the number of confirmed cases
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
    country_recovery_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
    country_active_cases.append(country_confirmed_cases[i] - country_death_cases[i] - country_recovery_cases[i])
    country_mortality_rate.append(country_death_cases[i]/country_confirmed_cases[i])
country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': country_confirmed_cases,
                          'Number of Deaths': country_death_cases, 'Number of Recoveries' : country_recovery_cases, 
                          'Number of Active Cases' : country_active_cases,
                          'Mortality Rate': country_mortality_rate})
# number of cases per country/region

country_df.style.background_gradient(cmap='Greens')
import requests
india_data_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()
df_india = pd.io.json.json_normalize(india_data_json['data']['statewise'])
df_india = df_india.set_index("state")

total = df_india.sum()
total.name = "Total"
df_t = pd.DataFrame(total,dtype=float).transpose()
df_t["Mortality Rate (per 100)"] = np.round(100*df_t["deaths"]/df_t["confirmed"],2)
df_india["Mortality Rate (per 100)"]= np.round(np.nan_to_num(100*df_india["deaths"]/df_india["confirmed"]),2)
df_india.style.background_gradient(cmap='Blues',subset=["confirmed"])\
                        .background_gradient(cmap='Reds',subset=["deaths"])\
                        .background_gradient(cmap='Greens',subset=["recovered"])\
                        .background_gradient(cmap='Purples',subset=["active"])\
                        .background_gradient(cmap='plasma',subset=["Mortality Rate (per 100)"])
import IPython
IPython.display.HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
## https://api.covid19india.org/csv/latest/state_wise.csv
    
# read data
latest = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')

# save as a .csv file`
latest.to_csv('state_level_latest.csv', index=False)

# first few rows
latest.head()
# read data
states_latest = pd.read_csv('state_level_latest.csv')

# remove the row containing 'Total'
states_latest = states_latest[states_latest['State']!='Total']

# mortality rate
states_latest['Mortality Rate'] = round((states_latest['Deaths'] / states_latest['Confirmed'])*100, 2)

# recovery rate
states_latest['Recovery Rate'] = round((states_latest['Recovered'] / states_latest['Confirmed'])*100, 2)

# first few rows
states_latest.head()
# read data
tests_state_wise = pd.read_csv('https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv')

# save as .csv file
tests_state_wise.to_csv('tests_state_wise.csv', index=False)

# first few rows
tests_state_wise.head(3)
# read data
tests_day_wise = pd.read_csv('https://api.covid19india.org/csv/latest/tested_numbers_icmr_data.csv')

# save as .csv file
tests_day_wise.to_csv('tests_day_wise.csv', index=False)

# first few rows
tests_day_wise.head()
# tests day by day
# read data
tests_dbd = pd.read_csv('tests_day_wise.csv')

# to datetime datatype
tests_dbd['Update Time Stamp'] = tests_dbd['Update Time Stamp'].str.extract('(\d{1,2}/\d{1,2}/\d{4})')
tests_dbd['Update Time Stamp'] = pd.to_datetime(tests_dbd['Update Time Stamp'], format='%d/%m/%Y')

# first few rwos
tests_dbd.head()
# Latest Data
# sort based on confirmed cases
latest = states_latest.sort_values('Confirmed', ascending=False).reset_index(drop=True)

# remove 'State Unassigned' row
latest = latest[latest['State']!='State Unassigned']

# rearrange columns
latest = latest.loc[:, ['State', 'Confirmed', 'Active', 'Deaths', 'Mortality Rate', 
                        'Recovered', 'Recovery Rate']]

# background color
latest.style\
    .background_gradient(cmap="Blues", subset=['Confirmed', 'Active'])\
    .background_gradient(cmap="Greens", subset=['Deaths', 'Mortality Rate'])\
    .background_gradient(cmap="Reds", subset=['Recovered', 'Recovery Rate'])
# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# interactive plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# offline plotly
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
# subset columns
temp = latest[['Active', 'Deaths', 'Recovered']]

# rename columns
temp.columns = ['Active', 'Deaths', 'Cured']

# melt into longer format
tm = temp.melt(value_vars=['Active', 'Deaths', 'Cured'])

# plot
fig_1 = px.treemap(tm, path=["variable"], values="value", height=250, 
                 color_discrete_sequence=[act, rec, dth], title='Latest stats')
fig_1.data[0].textinfo = 'label+text+value'
fig_1.show()
# daily
# read data
daily = pd.read_csv('../input/covid19-india/nation_level_daily.csv')

# convert datetime datatype
daily['Date'] = pd.to_datetime(daily['Date'] + ' 2020')

# get no. of active cases
daily['Total Active'] = daily['Total Confirmed'] - daily['Total Deceased'] - daily['Total Recovered']

# Deaths / 100 cases
daily['Deaths / 100 Cases'] = round((daily['Total Deceased'] / daily['Total Confirmed'])*100, 2)
# Recovered / 100 cases
daily['Recovered / 100 Cases'] = round((daily['Total Recovered'] / daily['Total Confirmed'])*100, 2)
# Deaths / 100 recovered
daily['Deaths / 100 Recovered'] = round((daily['Total Deceased'] / daily['Total Recovered'])*100, 2)

# first few rows
daily.head()
def plot_daily(col, hue):
    fig = px.bar(daily, x="Date", y=col, title=col, 
                 color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title="", yaxis_title="")
    fig.show()
plot_daily('Total Confirmed', '#000000')
plot_daily('Daily Confirmed', '#000000')

plot_daily('Total Active', '#000000')
plot_daily('Total Deceased', dth)

plot_daily('Daily Deceased', dth)
plot_daily('Total Recovered', rec)
plot_daily('Daily Recovered', rec)
def plot_daily(col, hue):
    temp = tests_dbd.copy()
    # temp = temp[~temp[col].isna()]
    fig = px.scatter(temp, x="Update Time Stamp", 
                 y=col, title=col, 
                 color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title="", yaxis_title="")
    fig.show()
plot_daily('Total Samples Tested', dth)
# stacked bar chart

# subset columns
temp = daily[['Date', 'Total Active', 'Total Deceased', 'Total Recovered']]

# melt data
temp = temp.melt(value_vars=['Total Recovered', 'Total Deceased', 'Total Active'],
                 id_vars="Date", var_name='Case', value_name='Count')
temp.head()

# plot
fig_2 = px.bar(temp, x="Date", y="Count", color='Case', 
               title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig_2.show()
# stacked bar chart

# subset columns
temp = daily[['Date', 'Daily Confirmed', 'Daily Deceased', 'Daily Recovered']]

# melt data
temp = temp.melt(value_vars=['Daily Recovered', 'Daily Deceased', 'Daily Confirmed'],
                 id_vars="Date", var_name='Case', value_name='Count')
temp.head()

# plot
fig_2 = px.bar(temp, x="Date", y="Count", color='Case', 
               title='Daily cases over time', color_discrete_sequence = [rec, dth, act])
fig_2.show()
# Daily statistics
# ================

temp = daily[daily['Total Confirmed'] > 100]

fig_c = px.line(temp, x="Date", y="Deaths / 100 Cases", color_discrete_sequence=['#000000'])
fig_d = px.line(temp, x="Date", y="Recovered / 100 Cases", color_discrete_sequence=['#649d66'])
fig_r = px.line(temp, x="Date", y="Deaths / 100 Recovered", color_discrete_sequence=['#ff677d'])

fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 
                    subplot_titles=('No. of Deaths to 100 Cases', 
                                    'No. of Recovered to 100 Cases', 
                                    'No. of Deaths to 100 Recovered'))

fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)
fig.add_trace(fig_r['data'][0], row=1, col=3)
temp = daily.loc[:, ['Date', 'Total Active', 'Total Recovered']]
temp = temp.melt(id_vars='Date', value_vars=['Total Active', 'Total Recovered'])
temp.head()

fig_c = px.line(temp, x="Date", y="value", color='variable', line_dash='variable', 
                color_discrete_sequence=[dth, rec])
fig_c.update_layout(title='Active vs Recovered cases', 
                  xaxis_title='', yaxis_title='')
fig_c.show()
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
import plotly.offline as py
import plotly_express as px 
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
hotspots = ['China','Germany','Iran','Italy','Spain','US','Korea, South','France','Turkey','United Kingdom','India']
dates = list(confirmed_df.columns[4:])
dates = list(pd.to_datetime(dates))
dates_india = dates[8:]

df1 = confirmed_df.groupby('Country/Region').sum().reset_index()
df2 = deaths_df.groupby('Country/Region').sum().reset_index()
df3 = recovered_df.groupby('Country/Region').sum().reset_index()

global_confirmed = {}
global_deaths = {}
global_recovered = {}
global_active= {}
k = df1[df1['Country/Region']=='India'].loc[:,'1/22/20':]
india_confirmed = k.values.tolist()[0] 
data = pd.DataFrame(columns = ['ds','y'])
data['ds'] = dates
data['y'] = india_confirmed

prop=Prophet()
prop.fit(data)
future=prop.make_future_dataframe(periods=30)
prop_forecast=prop.predict(future)
forecast = prop_forecast[['ds','yhat']].tail(30)

fig = plot_plotly(prop, prop_forecast)
fig = prop.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
arima = ARIMA(data['y'], order=(5, 1, 0))
arima = arima.fit(trend='c', full_output=True, disp=True)
forecast = arima.forecast(steps= 30)
pred = list(forecast[0])

start_date = data['ds'].max()
prediction_dates = []
for i in range(30):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
plt.figure(figsize= (15,10))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Total cases',fontsize = 20)
plt.title("Predicted Values for the next 15 Days" , fontsize = 20)

plt.plot_date(y= pred,x= prediction_dates,linestyle ='dashed',color = '#ff9999',label = 'Predicted');
plt.plot_date(y=data['y'],x=data['ds'],linestyle = '-',color = 'blue',label = 'Actual');
plt.legend();
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
covid_19_india=pd.read_csv('../input/covid19-india/covid_19_india.csv')

covid_19_india['Date']=pd.to_datetime(covid_19_india['Date'])
covid_19_india.head()
covid_19_india['ConfirmedForeignNational'].replace('-',0,inplace=True)
covid_19_india['ConfirmedIndianNational'].replace('-',0,inplace=True)
covid_19_india['ConfirmedIndianNational']=covid_19_india['ConfirmedIndianNational'].astype('int64')
covid_19_india['ConfirmedForeignNational']=covid_19_india['ConfirmedForeignNational'].astype('int64') 
lbl=LabelEncoder()
covid_19_india['State/UnionTerritory']=lbl.fit_transform(covid_19_india['State/UnionTerritory'])
covid_19_india['date']=covid_19_india['Date'].dt.day
covid_19_india['month']=covid_19_india['Date'].dt.month
x=covid_19_india[['State/UnionTerritory','date','month','Cured','Deaths','ConfirmedIndianNational','ConfirmedForeignNational']]
y=covid_19_india['Confirmed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split 
lm = LinearRegression()
lm.fit(x_train, y_train )
print(lm.coef_)
predictions = lm.predict(x_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
accuracy = regressor.score(x_test,y_test)
print(accuracy*100,'%')
from sklearn.tree import DecisionTreeRegressor 
tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
accuracy = tree.score(x_test,y_test)
print(accuracy*100,'%')