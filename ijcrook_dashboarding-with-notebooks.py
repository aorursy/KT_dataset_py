import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import datetime as dt  

import os



data = pd.read_csv('../input/procurement-notices.csv')





week_ago = dt.datetime.now() -dt.timedelta(days=7)

today = dt.datetime.now()

data['Publication Date'] = pd.to_datetime(data['Publication Date'])

data['Deadline Date'] = pd.to_datetime(data['Deadline Date'])

data_within_last_week  = data[data['Publication Date'] >= week_ago]

outstanding_data  = data[(data['Deadline Date'] >= today) | (pd.isna(data['Deadline Date']))]

outstanding_by_country = outstanding_data['Country Name'].value_counts().to_frame()

outstanding_by_country = outstanding_by_country.rename(columns={outstanding_by_country.columns[0]:"counts"})
country_counts_last_week = data_within_last_week.groupby('Country Name').count()['ID']

country_counts_last_week = country_counts_last_week.sort_values(ascending=False)[:50]

sector_counts_last_week = data_within_last_week.groupby('Major Sector').count()['ID']

sector_counts_last_week = sector_counts_last_week.sort_values(ascending=False)
import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Bar(x=country_counts_last_week.index, y=country_counts_last_week)]



# specify the layout of our figure

layout = dict(title = "Number of Contracts by Country Over the Last Week",

              xaxis= dict(title= 'Country',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
data = [ dict(

        type='choropleth',

        autocolorscale = False,

        locations = outstanding_by_country.index,

        z = outstanding_by_country['counts'],

        locationmode = 'country names',

        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        reversescale = True

       ) ]



# chart information

layout = dict(

        width= 800,

        height= 700,

        title = 'Oustanding Bids by Country',

        geo = dict(

        showframe = False,

        showcoastlines = False,

        showcountries=True,

        projection = dict(

            type = 'mercator'

        )))

   

# actually show our figure

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )