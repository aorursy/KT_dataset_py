import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import seaborn as sns
import plotly as py
import plotly.express as px

from fbprophet.plot import plot_plotly
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

import warnings
warnings.filterwarnings('ignore')
covid19confirmed = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_confirmed_global.csv')

covid19recovered = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_recovered_global.csv')

covid19deaths = pd.read_csv('../input/from-john-hopkins-university/time_series_covid19_deaths_global.csv')

covid19 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])

covid19Brazil = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')
#Checking the last 5 records to confirm when each dataset was updated:

print('covid19confirmed:')
print(covid19confirmed.tail())

###

print('covid19recovered:')
print(covid19recovered.tail())

###

print('covid19deaths:')
print(covid19deaths.tail())

###

print('covid19:')
print(covid19.tail())

###

print('covid19Brazil:')
print(covid19Brazil.tail())

###
#Rename column "ObservationDate" to 'Date'

covid19 = covid19.rename(columns={'ObservationDate' : 'Date'})
#Dataset dimensions (rows vs columns)

print('covid19confirmed:')
print(covid19confirmed.shape)

###

print('covid19recovered:')
print(covid19recovered.shape)

###

print('covid19deaths:')
print(covid19deaths.shape)

###

print('covid19:')
print(covid19.shape)

###

print('covid19Brazil:')
print(covid19Brazil.shape)

###
#Checking for null or missing data values in each dataset

print('covid19confirmed:')
print(pd.DataFrame(covid19confirmed.isnull().sum()))

###
print('covid19recovered:')
print(pd.DataFrame(covid19recovered.isnull().sum()))

###

print('covid19deaths:')
print(pd.DataFrame(covid19deaths.isnull().sum()))

###

print('covid19:')
print(pd.DataFrame(covid19.isnull().sum()))

###

print('covid19Brazil:')
print(pd.DataFrame(covid19Brazil.isnull().sum()))

###
#Some dataset have null or missings data values, then let's replace to "unknow" values

covid19confirmed = covid19confirmed.fillna('unknow') 
covid19recovered = covid19recovered.fillna('unknow')
covid19deaths = covid19deaths.fillna('unknow')
covid19 = covid19.fillna('unknow')
all_cases_world = covid19.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum()
all_cases_world = all_cases_world.reset_index()
all_cases_world = all_cases_world.sort_values('Date', ascending=False)

fig = go.Figure()
fig.update_layout(title_text='Total number of confirmed, deaths and recovered cases in the World', 
                  xaxis_title='Period Date', yaxis_title='Total Cases', template='plotly_dark')

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Confirmed'],
                        mode='lines+markers',
                        name='Global Confirmed',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Deaths'],
                        mode='lines+markers',
                        name='Global Deaths',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=all_cases_world['Date'],
                        y=all_cases_world['Recovered'],
                        mode='lines+markers',
                        name='Global Recovered',
                        line=dict(color='green', width=2)))


fig.show()
global_rate = covid19.groupby(['Date']).agg({'Confirmed':['sum'],'Deaths':['sum'], 'Recovered': ['sum']})
global_rate.columns = ['Global_Confirmed', 'Global_Deaths', 'Global_Recovered']
global_rate = global_rate.reset_index()
global_rate['Increase_New_Cases_by_Day'] = global_rate['Global_Confirmed'].diff().shift(-1)

#Calculating rates
#Lambda function
global_rate['Global_Deaths_rate_%'] = global_rate.apply(lambda row: ((row.Global_Deaths)/(row.Global_Confirmed))*100, axis=1).round(2)
global_rate['Global_Recovered_rate_%'] = global_rate.apply(lambda row: ((row.Global_Recovered)/(row.Global_Confirmed))*100, axis=1).round(2)
global_rate['Global_Growth_rate_%'] = global_rate.apply(lambda row: row.Increase_New_Cases_by_Day/row.Global_Confirmed*100, axis=1).round(2)
global_rate['Global_Growth_rate_%'] = global_rate['Global_Growth_rate_%'].shift(+1)

fig = go.Figure()
fig.update_layout(title_text='Global rate of growth confirmed, deaths and recovered cases',
                 xaxis_title='Period Date', yaxis_title='Rate', template='plotly_dark')

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Growth_rate_%'],
                        mode='lines+markers',
                        name='Global Growth Confirmed rate %',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Deaths_rate_%'],
                        mode='lines+markers',
                        name='Global Deaths rate %',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=global_rate['Date'],
                        y=global_rate['Global_Recovered_rate_%'],
                        mode='lines+markers',
                        name='Global Recovered rate %',
                        line=dict(color='green', width=2)))

fig.show()
global_rate.loc[:,['Date','Global_Confirmed', 'Global_Deaths', 'Global_Recovered', 'Increase_New_Cases_by_Day']].tail()
last_update = '4/20/20'
global_cases = covid19confirmed
global_cases = global_cases[['Country/Region', last_update]]
global_cases = global_cases.groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)
global_cases['Confirmed'] = covid19confirmed[['Country/Region', last_update]].groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)
global_cases['Recovered'] = covid19recovered[['Country/Region', last_update]].groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)
global_cases['Deaths'] = covid19deaths[['Country/Region', last_update]].groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)
global_cases['Active'] = global_cases[last_update] - global_cases['Recovered'] - global_cases['Deaths']
global_cases['Mortality_Rate_%'] = ((global_cases['Deaths'])/(global_cases['Confirmed'])*100).round(2)
global_cases = global_cases.loc[:,['Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality_Rate_%']]
global_cases.head(50)
prediction = covid19.copy()

prediction = prediction.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']})
prediction.columns = ['Confirmed', 'Deaths', 'Recovered']
prediction = prediction.reset_index()
prediction = prediction[prediction.Confirmed!=0]
prediction = prediction[prediction.Deaths!=0]

#Prevent division by zero
def ifNull(d):
    temp=1
    if d!=0:
        temp=d
    return temp

prediction['mortality_rate'] = prediction.apply(lambda row: ((row.Deaths+1)/ifNull((row.Confirmed)))*100, axis=1)
floorVar = 0
worldPop = 10000000

#Modelling total confirmed cases 
confirmed_train_dataset = pd.DataFrame(covid19.groupby('Date')['Confirmed'].sum().reset_index()).rename(columns={'Date': 'ds', 'Confirmed': 'y'})
confirmed_train_dataset['floor'] = floorVar
confirmed_train_dataset['cap'] = worldPop

#Modelling deaths
deaths_train_dataset = pd.DataFrame(covid19.groupby('Date')['Deaths'].sum().reset_index()).rename(columns={'Date': 'ds', 'Deaths': 'y'})
deaths_train_dataset['floor'] = 0
deaths_train_dataset['cap'] = 2500

#Modelling mortality rate
mortality_train_dataset = pd.DataFrame(prediction.groupby('Date')['mortality_rate'].mean().reset_index()).rename(columns={'Date': 'ds', 'mortality_rate': 'y'})
#Total dataframe model
m = Prophet(
    growth="logistic",
    interval_width=0.98,
    yearly_seasonality=False,
    weekly_seasonality=False,
    seasonality_mode='additive')

m.fit(confirmed_train_dataset)
future = m.make_future_dataframe(periods=50)
future['cap'] = worldPop
future['floor'] = floorVar
confirmed_forecast = m.predict(future)

#Mortality rate model
m_mortality = Prophet()
m_mortality.fit(mortality_train_dataset)
mortality_future = m_mortality.make_future_dataframe(periods=31)
mortality_forecast = m_mortality.predict(mortality_future)

#Deaths model
m2 = Prophet(
    growth="logistic",
    interval_width=0.95)
m2.fit(deaths_train_dataset)
future2 = m2.make_future_dataframe(periods=7)
future2['cap'] = 2500
future2['floor'] = 0
deaths_forecast = m2.predict(future2)
fig = plot_plotly(m, confirmed_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Total predictions to Confirmed cases in the World',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig.update_layout(annotations=annotations)
fig
fig_deaths = plot_plotly(m2, deaths_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Total predictions to Deaths in the World',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig_deaths.update_layout(annotations=annotations)
fig_deaths
fig_lethality = plot_plotly(m_mortality, mortality_forecast)
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                       xanchor='left', yanchor='bottom',
                       text='Predictions of the lethality rate in the World',
                       font=dict(family='Arial',
                                size=25,
                                color='rgb(37,37,37)'),
                       showarrow=False))
fig_lethality.update_layout(annotations=annotations)
fig_lethality
Brazil_cases = covid19.copy()
Brazil_cases = covid19.loc[covid19['Country/Region']=='Brazil']
Brazil_cases = Brazil_cases.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']}).sort_values('Date', ascending=False)
Brazil_cases.columns = ['Confirmed', 'Deaths', 'Recovered']
Brazil_cases = Brazil_cases.reset_index()
Brazil_cases['Confirmed_New_Daily_Cases'] = Brazil_cases['Confirmed'].diff().shift(-1)
Brazil_cases['Deaths_New_Daily_Cases'] = Brazil_cases['Deaths'].diff().shift(-1)
Brazil_cases['Recovered_New_Daily_Cases'] = Brazil_cases['Recovered'].diff().shift(-1)
Brazil_cases_confirmed = Brazil_cases[Brazil_cases['Confirmed']!=0]
#Brazil_cases_confirmed
fig = go.Figure()
fig.update_layout(title_text='Confirmed, Deaths and Recoveries cases in Brazil',
                 xaxis_title='Period Date', yaxis_title='Cases', template='plotly_dark')

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Confirmed'],
                        mode='lines+markers',
                        name='Brazil Confirmed Cases',
                        line=dict(color='yellow', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Deaths'],
                        mode='lines+markers',
                        name='Brazil Deaths Cases',
                        line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=Brazil_cases_confirmed['Date'],
                        y=Brazil_cases_confirmed['Recovered'],
                        mode='lines+markers',
                        name='Brazil Recovered Cases',
                        line=dict(color='green', width=2)))

fig.show()
Brazil_cases_rate = covid19.copy()
Brazil_cases_rate = covid19.loc[covid19['Country/Region']=='Brazil']
Brazil_cases_rate = Brazil_cases.groupby(['Date', 'Country/Region']).agg({'Confirmed':['sum'], 'Deaths':['sum'], 'Recovered':['sum']}).sort_values('Date', ascending=False)
Brazil_cases_rate.columns = ['Confirmed', 'Deaths', 'Recovered']
Brazil_cases_rate = Brazil_cases_rate.reset_index()
Brazil_cases_rate['Confirmed_New_Daily_Cases'] = Brazil_cases_rate['Confirmed'].diff().shift(-1)
Brazil_cases_rate = Brazil_cases_rate[Brazil_cases_rate.Confirmed!=0]
Brazil_cases_rate = Brazil_cases_rate[Brazil_cases_rate.Deaths!=0]

#Prevent division by zero

