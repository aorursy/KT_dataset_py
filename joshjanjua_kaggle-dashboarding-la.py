# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

import pandas as pd

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING



import seaborn as sns

import datetime



import datetime as DT



#Validator

import sys

!{sys.executable} -m pip install csvvalidator

import sys

import csv

from csvvalidator import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('../input/parking-citations.csv', low_memory=False)

# Any results you write to the current directory are saved as output.
#Validate data

field_names = (

               'Ticket number',

               'Issue Date',

               'Latitude',

               'Longitude',

               'Issue time',

               'Violation Description'

               )

validator = CSVValidator(field_names)

# basic header and record length checks

validator.add_header_check('EX1', 'bad header')

validator.add_record_length_check('EX2', 'unexpected record length')
#checking Issue Date format

assert False not in df['Issue Date'].str.match(r"^[\d]{4}-[\d]{2}-[\d]{2}T.*$").values

print('Issue Date is OK')



#violation Description

assert pd.api.types.is_string_dtype(df['Violation Description'])

print('Violation Description is OK')



#ticket number

assert pd.api.types.is_string_dtype(df['Ticket number'])

print('Ticket number is OK')



#Lat/Lon

assert pd.api.types.is_float_dtype(df['Latitude'])

print('Latitude is OK')

assert pd.api.types.is_float_dtype(df['Longitude'])

print('Longitude is OK')



#Issue Time

assert pd.api.types.is_float_dtype(df['Issue time'])

print('Issue time is OK')
#updating formatting so that I can translate issue date to datetime

df['Issue Date'] = df[df['Issue Date'].notnull()]['Issue Date'].apply(lambda x: x.split('T')[0])

df['Issue Date'] = pd.to_datetime(df['Issue Date'], infer_datetime_format=True)



#limiting dataset so it's easy to work with

today = pd.Timestamp('today').normalize()

week_ago = today - DT.timedelta(days=7)

month_ago = today - DT.timedelta(days=30)

df = df[df['Issue Date']>=month_ago]
#pad anything that is less than 4 digits then isolate just the hours

df['Issue time'] = df['Issue time'].astype(str)

df['Issue time'] = df['Issue time'].apply(lambda x: x.split('.')[0])

df['Issue time'] = df[df['Issue time'].notnull()]['Issue time'].apply(lambda x: x.zfill(4))

df['Issue Hour'] = df[df['Issue time']!='0nan']['Issue time'].apply(lambda x: DT.datetime.strptime(x,'%H%M').hour)



#clean lat lon

df['Latitude'] = np.where(df['Latitude']==99999.000, np.nan, df['Latitude'])

df['Longitude'] = np.where(df['Longitude']==99999.000, np.nan, df['Longitude'])



#string for ticketnum

df['Ticket number'] = df['Ticket number'].astype(str)
#Updating the Lat Lon

import pyproj

pm = '+proj=lcc +lat_1=34.03333333333333 +lat_2=35.46666666666667 +lat_0=33.5 +lon_0=-118 +x_0=2000000 +y_0=500000.0000000002 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs'

x1m,y1m = df['Latitude'].values, df['Longitude'].values

x2m,y2m = pyproj.transform(pyproj.Proj(pm,preserve_units = True), pyproj.Proj("+init=epsg:4326"), x1m,y1m)

df['Latitude']=x2m

df['Longitude']=y2m
import folium



LA_COORDINATES = (34.05, -118.24)



# create empty map zoomed in on San Francisco

map = folium.Map(location=LA_COORDINATES, zoom_start=10) 



# add a marker for every record in the filtered data, use a clustered view

from folium.plugins import FastMarkerCluster

FastMarkerCluster(data=list(zip(df[(df['Issue Date']>week_ago) & (df['Longitude'].notnull())]['Longitude'],(df[(df['Issue Date']>week_ago) & (df['Latitude'].notnull())]['Latitude'])))).add_to(map)



folium.LayerControl().add_to(map)

    

display(map)
#plot out scatter (line) graph of number of tickets 



df.set_index(df["Issue Date"],inplace=True)



#Creating the plot

data = [go.Scatter(x=df['Ticket number'].resample('D').count().truncate(before=month_ago).index, y=df['Ticket number'].resample('D').count().truncate(before=month_ago))]



# specify the layout of our figure

layout = dict(title = "Daily Number of Incidents",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
#line chart of the counts by time during the last month

data = [go.Scatter(x=df[['Ticket number','Issue Hour']].groupby('Issue Hour').count().index, y=df[['Ticket number','Issue Hour']].groupby('Issue Hour').count()['Ticket number'])]



# specify the layout of our figure

layout = dict(title = "Time of Incidents",

              xaxis= dict(title= 'Hour',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
#Top 10 reason codes during the last month

data = [go.Bar(x=df.groupby('Violation Description')['Ticket number'].count().sort_values(ascending = False)[:10].index, y=df.groupby('Violation Description')['Ticket number'].count().sort_values(ascending = False)[:10])]



# specify the layout of our figure

layout = dict(title = "Violations by Reason",

              xaxis= dict(ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)