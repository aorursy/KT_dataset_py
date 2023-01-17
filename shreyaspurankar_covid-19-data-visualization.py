#importing in all the libraries 

import numpy as np 

from datetime import date

import pandas as pd 

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

#loading in the dataset 

#this data set is provided by Kaggle 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading the dataset file

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#just some data cleaning

df = df.rename(columns={'Country/Region':'Country'}) #since it has included some provinces from Mainland China

df = df.rename(columns={'ObservationDate':'Date'}) #Just for aesthetics

df_countries = df.groupby(['Country', 'Date']).max().reset_index().sort_values('Date', ascending=False)

df_countries = df_countries.drop_duplicates(subset = ['Country'])

df_countries = df_countries[df_countries['Confirmed']>0] #considering only countries with existing cases

df_countrydate = df[df['Confirmed']>0]

df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()
#creating a choropleth map

#i have a timeseries data which im gonna animate



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

        showcoastlines = True,

    ))

    

fig.show()
fig = px.choropleth(df_countrydate, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Deaths", 

                    hover_name="Country", 

                    animation_frame="Date"

                   )

fig.update_layout(

    title_text = "COVID 19 - Deaths over the dates" ,

    title_x = 0.5, #alignment of title

    geo=dict(

        showframe = False,

        showcoastlines = False,

    )

    )

    

fig.show()
#the classic pie chart

fig = px.pie(df_countries, values = 'Confirmed',names='Country', height=600)

fig.update_traces(textposition='inside', textinfo='percent+label')



fig.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



fig.show()
#this one looked attractive to me, it gives you visualization of the comparative magnitude of numbers

fig = px.treemap(df_countries, 

                 path=['Country'],

                 values = 'Confirmed',

                 names='Country',

                 height=600,

                 title='Proportion of cases',

                )



fig.show()
#growth visualization

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
df_india = df_countries[df_countries['Country'] == 'India']

df_india
corona_india=df[df['Country']=='India']

line_data = corona_india.groupby('Date').sum().reset_index()



line_data = line_data.melt(id_vars='Date', 

                 value_vars=['Confirmed', 

                             'Recovered', 

                             'Deaths'], 

                 var_name='Ratio', 

                 value_name='Value')



fig = px.line(line_data, x="Date", y="Value", color='Ratio', 

              title='Confirmed cases, Recovered cases, and Death Over Time')

fig.show()
df[df['Country'] == 'India']