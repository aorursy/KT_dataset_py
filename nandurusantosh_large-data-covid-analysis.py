import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



import plotly.graph_objects as go

import pycountry

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df= pd.read_csv('/kaggle/input/large-data-covid/complete_data_new_format.csv',parse_dates=['Date'])
df.rename(columns={ 'Province/State':'State','Country/Region':'Country',}, inplace=True)

df['Country'] = df['Country'].replace('Mainland China', 'China')

df['Active'] = df['Confirmed'] - df['Deaths'] 

df['State'] = df['State'].fillna('')
grouped = df.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



fig = px.line(grouped, x='Date', y='Confirmed', 

              title='Confirmed Cases Over Time - Worldwide', color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()



fig = px.line(grouped, x='Date', y='Confirmed', 

              title='Confirmed Cases (Logarithmic Scale) Over Time- Worldwide', 

              log_y=True,color_discrete_sequence=px.colors.qualitative.Dark2)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()


italy = df[df['Country'] == 'Italy'].reset_index()

italy_date = italy.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



us = df[df['Country'] == 'US'].reset_index()

us_date = us.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()



india = df[df['Country'] == 'India'].reset_index()

india_date = india.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()





rest = df[~df['Country'].isin(['China', 'Italy', 'US','India'])].reset_index()

rest_date = rest.groupby('Date')['Date', 'Confirmed', 'Deaths'].sum().reset_index()


fig = px.line(italy_date,x='Date', y='Confirmed', 

              title="Confirmed Cases in Italy Over Time", 

              color_discrete_sequence=px.colors.qualitative.Dark2,

              height=500

             )

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()



fig = px.line(us_date, x='Date', y='Confirmed', 

              title="Confirmed Cases in USA Over Time", 

              color_discrete_sequence=px.colors.qualitative.Dark2,

              height=500

             )

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()



fig = px.line(india_date,x='Date', y='Confirmed', 

              title="Confirmed Cases in India Over Time", 

              color_discrete_sequence=px.colors.qualitative.Dark2,

              height=500

             )

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()



fig = px.line(rest_date, x='Date', y='Confirmed', 

              title="Confirmed Cases in Rest of the World Over Time", 

              color_discrete_sequence=px.colors.qualitative.Dark2,

              height=500

             )

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
confirmed = df[df['Date'] == max(df['Date'])].reset_index()

confirmedcases_grouped = confirmed.groupby('Country')['Confirmed', 'Deaths'].sum().reset_index()
fig = px.choropleth(confirmedcases_grouped, locations='Country', 

                    locationmode='country names', color='Confirmed', 

                    hover_name='Country', range_color=[1,5000], 

                    color_continuous_scale="algae", 

                    title='Countries with Confirmed Cases')



fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus'])



europedata= df[df['Country'].isin(europe)]

fig = px.choropleth(europedata, locations='Country', 

                    locationmode='country names', color='Confirmed', 

                    hover_name='Country', range_color=[1,2000], 

                    color_continuous_scale='algae', 

                    title='European Countries with Confirmed Cases', scope='europe', height=500)

fig.show()

df2= pd.read_csv('/kaggle/input/large-data-covid-2/covid_19_clean_complete.csv',parse_dates=['Date'])
df2.rename(columns={'Province/State':'State', 'Country/Region':'Country'}, inplace=True)
df2['State'] = df2['State'].fillna('')
df2_grouped= df2.groupby(['Date', 'Country'])['Confirmed', 'Deaths'].max().reset_index()

df2_grouped['Date'] = pd.to_datetime(df2_grouped['Date'])

df2_grouped['Date'] = df2_grouped['Date'].dt.strftime('%m/%d/%Y')

df2_grouped['Size'] = df2_grouped['Confirmed'].pow(0.5)



fig = px.scatter_geo(df2_grouped, locations='Country', locationmode='country names', 

                     color='Confirmed', size='Size', hover_name='Country', 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Spread Over Time', color_continuous_scale="algae")

fig.show()
