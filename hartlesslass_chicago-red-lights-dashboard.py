# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import plotly.plotly as py

import plotly.graph_objs as go

import folium



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



cam_data=pd.read_csv("../input/red-light-camera-violations.csv")

cam_locations = pd.read_csv("../input/red-light-camera-locations.csv")
#Convert the violation date to a datetime format for use in time series analysis

cam_data['VIOLATION DATE']=pd.to_datetime(cam_data['VIOLATION DATE'])

#Get the most recent date in the dataset. This will be used to create dynamic titles for the charts

max_date = cam_data['VIOLATION DATE'].max()

#set the range of days to look back for the report

days_to_subtract=30

#find the date x number of days prior to the most recent date in the dataset

d = max_date - timedelta(days=days_to_subtract)



#Create a subset of the cam_data to only include the number of days selected

last_thirty=cam_data[cam_data['VIOLATION DATE']>=d]



#summarize the violations over the subset time period

total_violations = last_thirty.groupby('VIOLATION DATE')[['VIOLATIONS']].agg('sum')

total_violations['date']=total_violations.index

total_violations=total_violations.reset_index(drop=True)



#Create a function which will define each of the weekend nights (friday & saturday) and mark them with a vertical rectangle on the graph

def shape(startdate, enddate):

    shapes=[]

    while startdate <= enddate:

        #determine if the day of week is a Friday or Saturday

        if startdate.weekday() ==4:

            shapes.append({

            'type': 'rect',

            'xref': 'x',

            'yref': 'paper',

            'x0': startdate,

            'y0': 0,

            'x1': startdate+timedelta(days=1),

            'y1': 1,

            'fillcolor': '#d3d3d3',

            'opacity': 0.3,

            'line': {

                'width': 0,

                }

            }

            )

            #skip a date if the day was a Friday

            startdate=startdate+timedelta(days=1)

        startdate=startdate+timedelta(days=1)

    return shapes



#plot the time series data

data = [go.Scatter(x=total_violations['date'], y=total_violations['VIOLATIONS'])]



#layout = dict(title = 'Number of Violations between {} and {}'.format(d,max_date),

#             xaxis= dict(title='Violations', ticklen=1, zeroline=False))

layout = {'title':'Number of Violations between {:%x} and {:%x}'.format(d,max_date),

    # to highlight the timestamp we use shapes and create a rectangular

    'shapes': shape(d,max_date)}



fig = dict(data=data, layout=layout)

iplot(fig)
#Get the top 10 locations with the highest Violations during the select time period

top_ten=last_thirty.copy()

top_ten = top_ten.groupby([top_ten['INTERSECTION'],top_ten['LATITUDE'],top_ten['LONGITUDE']])[['VIOLATIONS']].agg('sum')

top_ten[['INTERSECTION', 'LATITUDE', 'LONGITUDE']]=pd.DataFrame(top_ten.index.tolist(), index=top_ten.index) 

top_ten=top_ten.reset_index(drop=True)



top_ten=top_ten.sort_values(['VIOLATIONS'], ascending=False).head(10)

#make the x axis labels wrap

x=top_ten['INTERSECTION'].str.replace(" ","<br>")

data = [go.Bar(

            x=x,

            y=top_ten['VIOLATIONS']

    )]

layout = go.Layout(title='Top 10 Violation Locations between {:%x} and {:%x}'.format(d,max_date),)

fig=go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')

#map the top 10 locations by total violations during the time period onto a map of Chicago

chicago_location = [41.8781, -87.6298]



m = folium.Map(location=chicago_location, zoom_start=12)

for i in range(0,len(top_ten)):

    folium.Circle(

      location=[top_ten.iloc[i]['LATITUDE'], top_ten.iloc[i]['LONGITUDE']],

      popup="{}: {}".format(top_ten.iloc[i]['INTERSECTION'],top_ten.iloc[i]['VIOLATIONS']),

      radius=int(top_ten.iloc[i]['VIOLATIONS']),

      color='crimson',

      fill=True,

      fill_color='crimson'

    ).add_to(m)

m