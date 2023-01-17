import numpy as np 

import pandas as pd 

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df = df.rename(columns={'Country/Region':'Country'})

df = df.rename(columns={'ObservationDate':'Date'})

df.tail()
df.describe()

df['Date'].max()
df_countries = df.groupby(['Country', 'Date']).max().reset_index().sort_values('Date', ascending=False)

df_countries = df_countries.drop_duplicates(subset = ['Country'])

df_countries = df_countries[df_countries['Confirmed']>0]



df_countries
fig = px.pie(df_countries, values = 'Confirmed',names='Country', height=600)

fig.update_traces(textposition='inside', textinfo='percent+label')



fig.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



fig.show()
df_countrydate = df[df['Confirmed']>0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()

df_countrydate
fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )



fig.update_layout(

    title_text = 'Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()
fig = px.treemap(df_countries, 

                 path=['Country'],

                 values = 'Confirmed',

                 names='Country',

                 height=600,

                 title='Proportion of cases',

                )



fig.show()
bar_data = df.groupby(['Country', 'Date'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values('Date', ascending=True)



fig = px.bar(bar_data, x="Date", y="Confirmed", color='Country', text = 'Confirmed', orientation='v', height=600,

             title='Cases')

fig.show()



fig = px.bar(bar_data, x="Date", y="Deaths", color='Country', text = 'Deaths', orientation='v', height=600,

             title='Deaths')

fig.show()



fig = px.bar(bar_data, x="Date", y="Recovered", color='Country', text = 'Recovered', orientation='v', height=600,

             title='Recovered')

fig.show()
line_data = df.groupby('Date').sum().reset_index()



line_data = line_data.melt(id_vars='Date', 

                 value_vars=['Confirmed', 

                             'Recovered', 

                             'Deaths'], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(line_data, x="Date", y="Value", color='Ratio', 

              title='Confirmed cases, Recovered cases, and Death Over Time')

fig.show()