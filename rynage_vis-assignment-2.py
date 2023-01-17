import numpy as np

import pandas as pd

from datetime import date, timedelta



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
covid = pd.read_csv('../input/owid-covid-data.csv', parse_dates=[3])

covid.date = pd.to_datetime(covid.date).dt.date
for locations in covid.location.unique():

    covid_at_location = covid[covid.location==locations]

    thedate = min(covid_at_location.date)

    maxdate = max(covid.date)

    while (thedate < maxdate):

        thedate = thedate + timedelta(1)

        if (not any(covid_at_location.date == thedate)):

            newentry = covid_at_location[covid_at_location.date == (thedate - timedelta(1))]

            newentry.date = newentry.date + timedelta(1)

            newentry.new_cases = 0

            newentry.new_deaths = 0

            newentry.new_cases_per_million = 0

            newentry.new_deaths_per_million = 0

            if (newentry.new_tests.notna().any()):

                newentry.new_tests = 0

            if (newentry.new_tests_per_thousand.notna().any()):

                newentry.new_tests_per_thousand = 0

            covid = covid.append(newentry, ignore_index=True)

            covid_at_location = covid[covid.location==locations]

            #print(locations, thedate)

            #print(newentry)

            
covid.to_csv('./adjusted_covid.csv')
covid['total_cases'] = covid['total_cases'].astype(int)

covid_updated = covid[covid.date==max(covid.date)]

covid_updated = covid_updated[covid_updated.location!='World']



fig = px.scatter(covid_updated.sort_values('total_cases', ascending=False).iloc[:50, :], 

                 x='total_cases_per_million', y='total_deaths_per_million',log_x=True, log_y=True, 

                 color='location', height=700, size='total_cases', text='location',

                labels={'total_cases_per_million':'Total cases per million', 'total_deaths_per_million':'Total deaths per million'}, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
covid['total_cases'] = covid['total_cases'].astype(int)

covid_updated = covid[covid.date==max(covid.date)]

covid_updated = covid_updated[covid_updated.location!='World']



fig = px.scatter(covid_updated.sort_values('total_cases', ascending=False).iloc[:20, :], 

                 x='total_cases', y='total_deaths',log_x=True, log_y=True, 

                 color='location', height=700, size='total_cases', text='location',

                labels={'total_cases':'Confirmed', 'total_deaths':'Death'})

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
covid['total_cases'] = covid['total_cases'].astype(int)

covid_updated = covid[covid.date==max(covid.date)]

covid_updated = covid_updated[covid_updated.location!='World']



fig = px.scatter(covid_updated.sort_values('total_cases', ascending=False).iloc[:50,:], 

                 x='total_cases', y='total_deaths',log_x=True, log_y=True, 

                 color='location', height=700, size='total_cases', text='location',

                labels={'total_cases':'Confirmed', 'total_deaths':'Death'}, size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()