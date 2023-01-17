# install calmap

#! pip install calmap
# essential libraries

import json

import random

from urllib.request import urlopen



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

#import calmap

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



# html embedding

from IPython.display import Javascript

from IPython.core.display import display

from IPython.core.display import HTML
# list files

# !ls ../input/novel-corona-virus-2019-dataset/
# Read data

all_confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

all_deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

all_recovered_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
# all_confirmed_df.head(10)

# all_deaths_df.head(10)

# all_recovered_df.head(10)
# Merging tables



dates = all_confirmed_df.columns[4:]



all_confirmed_df_melt = all_confirmed_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                                              value_vars=dates, var_name='Date', value_name='Confirmed')



all_deaths_df_melt = all_deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                                        value_vars=dates, var_name='Date', value_name='Deaths')



all_recovered_df_melt = all_recovered_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                                              value_vars=dates, var_name='Date', value_name='Recovered')



world_cases_df = pd.concat([all_confirmed_df_melt, all_deaths_df_melt['Deaths'], all_recovered_df_melt['Recovered']], 

                       axis=1, sort=False)



world_cases_df.head(10)
# Dataframe info

# world_cases_df.info()
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

world_cases_df['Active'] = world_cases_df['Confirmed'] - world_cases_df['Deaths'] - world_cases_df['Recovered']



# replacing Mainland china with just China

world_cases_df['Country/Region'] = world_cases_df['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

world_cases_df[['Province/State']] = world_cases_df[['Province/State']].fillna('')

world_cases_df[cases] = world_cases_df[cases].fillna(0)

world_cases_df['Date'] = pd.to_datetime(world_cases_df['Date'])
russia_cases_df = world_cases_df.loc[world_cases_df['Country/Region'] == 'Russia']

russia_cases_df.head(60)
russia_cases_df = russia_cases_df.loc[russia_cases_df['Confirmed'] > 0]
temp = russia_cases_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,

                 color_discrete_sequence=[rec, act, dth])

fig.show()
temp = russia_cases_df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
temp = russia_cases_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index() 

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             color_discrete_sequence=[cnf, dth, rec])

fig.update_layout(barmode='group')

fig.show()

temp = world_cases_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,

                 color_discrete_sequence=[rec, act, dth])

fig.show()
temp = world_cases_df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.show()
temp = world_cases_df.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().diff()

temp = temp.reset_index() 

temp = temp.melt(id_vars="Date", 

                 value_vars=['Confirmed', 'Deaths', 'Recovered'])



fig = px.bar(temp, x="Date", y="value", color='variable', 

             color_discrete_sequence=[cnf, dth, rec])

fig.update_layout(barmode='group')

fig.show()
rus_reg_cases_df = pd.read_csv('../input/covid19-russia-regions-cases/covid19-russia-cases.csv')

rus_reg_cases_df.tail(10)
# rus_reg_cases_df.info()
rus_reg_cases_df['Date'] = pd.to_datetime(rus_reg_cases_df['Date'], dayfirst=True)



# Fix the dataset bug

rus_reg_cases_df['Region/City'] = rus_reg_cases_df['Region/City'].astype('str').str.strip('\u200b') 



rus_reg_cases_df['Active'] = rus_reg_cases_df['Confirmed'] - rus_reg_cases_df['Deaths'] - rus_reg_cases_df['Recovered']

# rus_reg_cases_df.tail(100)
# Get current situation in regions

rus_latest = rus_reg_cases_df.groupby('Region/City').apply(lambda df: df.loc[df['Date'].idxmax()])

rus_latest = rus_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)

rus_latest = rus_latest[['Region/City'] + cases]
rus_latest.style.background_gradient(cmap='Reds')
temp = rus_reg_cases_df.groupby(['Date', 'Region/City'])['Confirmed'].max().reset_index()

temp = temp.sort_values('Confirmed', ascending=False)



px.line(temp, x="Date", y="Confirmed", color='Region/City', title='Cases Spread', height=600)
# Russia data

day_cases = ['Day-Confirmed', 'Day-Deaths', 'Day-Recovered']

rus_sum = rus_reg_cases_df.loc[rus_reg_cases_df['Region/City'] != 'Diamond Princess']

rus_sum = rus_sum.groupby('Date').apply(lambda df: df[day_cases].sum())

rus_sum['Country/Region'] = 'Russia'

rus_sum = rus_sum.groupby('Country/Region').apply(lambda df: df[day_cases].cumsum()).reset_index()

rus_sum['Country/Region'] = 'Russia'

first_day = rus_sum['Date'][0]

rus_sum['Days'] = rus_sum.groupby('Date').apply(lambda df: df['Date'] - first_day).reset_index(drop=True)

rus_sum = rus_sum.rename(columns={'Day-Confirmed': 'Confirmed', 'Day-Deaths': 'Deaths', 'Day-Recovered': 'Recovered'})



columns = rus_sum.columns

threshold = rus_sum['Confirmed'].max() + 500



def country_data(country):

    country_df = world_cases_df.loc[world_cases_df['Country/Region'] == country]

    country_df = country_df.loc[(country_df['Confirmed'] > 0) & (country_df['Confirmed'] <= threshold)].reset_index(drop=True)



    first_day = country_df['Date'][0]

    country_df['Days'] = country_df.groupby('Date').apply(lambda df: df['Date'] - first_day).reset_index(drop=True)

    country_df = country_df[columns]

    return country_df



italy_df = country_data('Italy')

spain_df = country_data('Spain')

iran_df = country_data('Iran')

germany_df = country_data('Germany')

# france_df = country_data('France')

# us_df = country_data('US')

temp = pd.concat([rus_sum, italy_df, spain_df, iran_df, germany_df])



temp['Days'] = temp['Days'].astype('str')

temp1 = temp.groupby(['Country/Region', 'Date']).apply(lambda df: int(df['Days'][0].split(' ')[0])).reset_index()

temp1 = temp1.sort_values(['Date', 'Country/Region']).reset_index(drop=True)

temp = temp.sort_values(['Date', 'Country/Region']).reset_index(drop=True)

temp['Days'] = temp1[0]



px.line(temp, x="Days", y="Confirmed", color='Country/Region', title='First 1000 Cases Spread', height=600)

moscow_cases = rus_reg_cases_df.loc[rus_reg_cases_df['Region/City'] == 'Москва']

moscow_cases
moscow_cases = moscow_cases.melt(id_vars="Date", 

                 value_vars=['Day-Confirmed', 'Day-Deaths', 'Day-Recovered'])



fig = px.bar(moscow_cases, x="Date", y="value", color='variable', 

             color_discrete_sequence=[cnf, dth, rec])

fig.update_layout(barmode='group')

fig.show()