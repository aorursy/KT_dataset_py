!pip install jovian --upgrade --quiet
projectName = 'covid19-data-visualization'

import pandas as pd

import matplotlib.pyplot as plt
import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objects as go

import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'kaggle'
covid_df = pd.read_csv('../input/covid19/covid.csv')

covid_grouped_df = pd.read_csv('../input/covid19/covid_grouped.csv')
covid_df
covid_grouped_df
covid_df.columns
covid_df.drop(['NewCases', 'NewDeaths', 'NewRecovered' ], axis = 1, inplace = True) # this are columns which are empty for all rows

#axis = 1 will delete columns

#implace = True will not create new dataset
covid_df
from plotly.figure_factory import create_table

table = create_table(covid_df.head(20), colorscale=[[0, '#000000'],

                                 [.5, '#80beff'],

                                 [1, '#cce5ff']],)

table
covid_df.columns
px.bar(covid_df.head(20), x = 'Country/Region', y = 'TotalCases', color = 'TotalCases', height = 500, hover_data = ['Country/Region', 'Continent'])
px.bar(covid_df.head(20), x = 'Country/Region', y = 'TotalDeaths', color = 'TotalDeaths', height = 500, hover_data = ['Country/Region', 'Continent'])
px.bar(covid_df.head(20), x = 'Country/Region', y = 'Deaths/1M pop', color = 'Deaths/1M pop', height = 500, hover_data = ['Country/Region', 'Continent'])
px.bar(covid_df.head(20), x = 'Country/Region', y = 'TotalRecovered', color = 'TotalRecovered', height = 500, hover_data = ['Country/Region', 'Continent'])
px.bar(covid_df.head(20), x = 'TotalTests', y = 'Country/Region', color = 'TotalTests',orientation='h' ,height = 500, hover_data = ['Country/Region', 'Continent'])
px.bar(covid_df, x = 'TotalTests', y = 'Continent', color = 'TotalTests',orientation='h' ,height = 500, hover_data = ['Country/Region', 'Continent'])
px.scatter(covid_df,x = 'Continent', y='TotalCases', hover_data = ['Country/Region','Continent'], color='TotalCases' ,size='TotalCases', size_max=80, log_y=True)
px.scatter(covid_df.head(50),x = 'Continent', y='TotalTests', hover_data = ['Country/Region','Continent'], color='TotalTests' ,size='TotalTests', size_max=80, log_y=True)
px.scatter(covid_df.head(50),x = 'Continent', y='TotalDeaths', hover_data = ['Country/Region','Continent'], color='TotalDeaths' ,size='TotalDeaths', size_max=80, log_y=True)
px.scatter(covid_df,x = 'Country/Region', y='TotalCases', hover_data = ['Country/Region','Continent'], color='Country/Region' ,size='TotalCases', size_max=100, log_y = True)
px.scatter(covid_df.head(20),x = 'Country/Region', y='TotalDeaths', hover_data = ['Country/Region','Continent'], color='TotalDeaths' ,size='TotalDeaths', size_max=80)
px.scatter(covid_df.head(40),x = 'Country/Region', y='Tests/1M pop', hover_data = ['Country/Region','Continent'], color='Tests/1M pop' ,size='Tests/1M pop', size_max=80)
px.scatter(covid_df.head(40),x = 'TotalCases', y='TotalDeaths', hover_data = ['Country/Region','Continent'], color='TotalDeaths' ,size='TotalDeaths', size_max=80)
px.scatter(covid_df.head(40),x = 'TotalCases', y='TotalDeaths', hover_data = ['Country/Region','Continent'], color='TotalDeaths' ,size='TotalDeaths', size_max=80,log_x=True, log_y=True)
px.scatter(covid_df.head(40),x = 'TotalTests', y='TotalCases', hover_data = ['Country/Region','Continent'], color='TotalCases' ,size='TotalCases', size_max=80,log_x=True, log_y=True)
covid_grouped_df.columns
covid_grouped_df
px.bar(covid_grouped_df, x = 'Date', y = 'Confirmed', color='Confirmed', hover_data=['Confirmed','Date','Country/Region'], height=500)
px.bar(covid_grouped_df, x = 'Date', y = 'Confirmed', color='Confirmed', hover_data=['Confirmed','Date','Country/Region'], height=500,log_y=True)
px.bar(covid_grouped_df, x = 'Date', y = 'Deaths', color='Deaths', hover_data=['Confirmed','Date','Country/Region'], height=500)
## India's data visualization
df_india = covid_grouped_df.loc[covid_grouped_df['Country/Region']=='India']

df_india
px.bar(df_india, x = 'Date', y = 'New cases', color='New cases', hover_data=['Confirmed','Date','Country/Region'], height=500)
px.line(df_india, x = 'Date', y = 'New cases', height=500)
px.line(df_india, x = 'Date', y = 'Confirmed', height=500)
px.bar(df_india, x = 'Date', y = 'Confirmed',color='Confirmed', height=500)
px.bar(df_india, x = 'Date', y = 'New deaths',color='New deaths', height=500)
px.bar(df_india, x = 'Date', y = 'Deaths',color='Deaths', height=500)
px.bar(df_india, x = 'Date', y = 'Recovered',color='Recovered', height=500)
px.bar(df_india, x = 'Date', y = 'New recovered',color='New recovered', height=500)
px.choropleth(covid_grouped_df,

                 locations='iso_alpha',

                 color='Confirmed',

                 hover_name='Country/Region',

                 color_continuous_scale='Reds',

                 animation_frame='Date')
px.choropleth(covid_grouped_df,

                 locations='iso_alpha',

                 color='Deaths',

                 hover_name='Country/Region',

                 color_continuous_scale='Viridis',

                 animation_frame='Date')
px.choropleth(covid_grouped_df,

                 locations='iso_alpha',

                 color='Deaths',

                 hover_name='Country/Region',

                 color_continuous_scale='Viridis',

                 projection = 'orthographic',

                 animation_frame='Date')
px.choropleth(covid_grouped_df,

                 locations='iso_alpha',

                 color='Recovered',

                 hover_name='Country/Region',

                 color_continuous_scale='RdYlGn',

                 projection = 'natural earth',

                 animation_frame='Date')
px.bar(covid_grouped_df,x='WHO Region', y='Confirmed', color='WHO Region', animation_frame='Date', hover_name='Country/Region')
px.bar(covid_grouped_df,x='WHO Region', y='New cases', color='WHO Region', animation_frame='Date', hover_name='Country/Region')
import jovian
jovian.commit(project='covid19-data-visualization')