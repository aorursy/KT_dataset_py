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

import bq_helper



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



cam_data=pd.read_csv("../input/red-light-camera-violations.csv")

cam_locations = pd.read_csv("../input/red-light-camera-locations.csv")

noaa_data_set = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",

                                        dataset_name= "noaa_gsod")
#Convert the violation date to a datetime format for use in time series analysis

cam_data['VIOLATION DATE']=pd.to_datetime(cam_data['VIOLATION DATE'])

#Get the most recent date in the dataset. This will be used to create dynamic titles for the charts

max_date = cam_data['VIOLATION DATE'].max()

min_date = cam_data['VIOLATION DATE'].min()

cam_df=cam_data.groupby(cam_data['VIOLATION DATE'])[['VIOLATIONS']].agg('sum')

cam_df['date']=cam_df.index

cam_df=cam_df.reset_index(drop=True)

cam_df['date']=pd.to_datetime(cam_df['date'])

#get weather data for O'Hare airport in Chicago

base_query = """

SELECT

    CAST(CONCAT(year,'-',mo,'-',da) AS date) AS date,

    temp,

    wdsp,

    max AS max_temp,

    min AS min_temp,

    prcp,

    sndp AS snow_depth,

    fog,

    rain_drizzle,

    snow_ice_pellets,

    hail,

    thunder,

    tornado_funnel_cloud

FROM

"""



where_clause = """

WHERE stn='725300'

"""

tables=[

    "`bigquery-public-data.noaa_gsod.gsod2019`",

    "`bigquery-public-data.noaa_gsod.gsod2018`",

    "`bigquery-public-data.noaa_gsod.gsod2017`",

    "`bigquery-public-data.noaa_gsod.gsod2016`",

    "`bigquery-public-data.noaa_gsod.gsod2015`",

    "`bigquery-public-data.noaa_gsod.gsod2014`"]



for t in range(len(tables)):

    if t==0:

        query = "{0} {1} {2} \n".format(base_query,tables[t],where_clause)

    else:

        query+="UNION ALL \n {0} {1} {2}".format(base_query,tables[t],where_clause)



weather_data= noaa_data_set.query_to_pandas_safe(query, max_gb_scanned=2.0)

weather_data['date']=pd.to_datetime(weather_data['date'])



#merge weather data with violation data

weather=weather_data.merge(cam_df, left_on='date', right_on= 'date')

weather = weather.rename(columns={'VIOLATIONS_x': 'speed_violations', 'VIOLATIONS_y': 'red_light_violations'})



#handle outliers





#replace snow depth equal to 999.9 with 0 (999.9 is used for missing values)

weather['snow_depth']=weather['snow_depth'].replace(999.9,0.0)

#remove outliers from max temp column

weather=weather.drop(weather[weather.max_temp==weather.max_temp.max()].index)

total_rows=len(weather)



weather['winter_weather']=np.where((weather['min_temp']<=35.0) & (weather['prcp']>0), 1, 0)

weather = weather.rename(columns={'VIOLATIONS': 'red_light_violations'})
#Create a function which will define each of the weekend nights (friday & saturday) and mark them with a vertical rectangle on the graph

def shape(startdate, enddate,opacity):

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

            'opacity': opacity,

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

data = [go.Scatter(x=cam_df['date'], y=cam_df['VIOLATIONS'])]



#layout = dict(title = 'Number of Violations between {} and {}'.format(d,max_date),

#             xaxis= dict(title='Violations', ticklen=1, zeroline=False))

layout = {'title':'Number of Red Light Violations between {:%x} and {:%x}'.format(min_date,max_date),

    # to highlight the timestamp we use shapes and create a rectangular

    'shapes': shape(min_date,max_date,0.3)}



fig = dict(data=data, layout=layout)

iplot(fig)
top_num=25

locations=cam_data.groupby(['INTERSECTION', 'LATITUDE', 'LONGITUDE'], as_index=False)[['VIOLATIONS']].agg('sum')

locations=locations.sort_values(by=['VIOLATIONS'], ascending=False)

locations=locations.head(top_num)



chicago_location = [41.8781, -87.6298]



m = folium.Map(location=chicago_location, zoom_start=11)

for i in range(0,len(locations)):

    folium.Circle(

      location=[locations.iloc[i]['LATITUDE'], locations.iloc[i]['LONGITUDE']],

      popup="{}: {}".format(locations.iloc[i]['INTERSECTION'],locations.iloc[i]['VIOLATIONS']),

      radius=int(locations.iloc[i]['VIOLATIONS'])/100,

      color='crimson',

      fill=True,

      fill_color='crimson'

    ).add_to(m)

m
data = [go.Scatter(x=weather['prcp'],

    y=weather['red_light_violations'],

    mode='markers')]



#layout = dict(title = 'Number of Violations between {} and {}'.format(d,max_date),

#             xaxis= dict(title='Violations', ticklen=1, zeroline=False))

layout = {'title':'Correlation between Precipitation and Red Light Violations',

          'xaxis': {'title':'Precipitation'},

          'yaxis': {'title': 'Red Light Violations'}

}



fig = dict(data=data, layout=layout)

iplot(fig)
data = [go.Scatter(x=weather['min_temp'],

    y=weather['red_light_violations'],

    mode='markers')]



#layout = dict(title = 'Number of Violations between {} and {}'.format(d,max_date),

#             xaxis= dict(title='Violations', ticklen=1, zeroline=False))

layout = {'title':'Correlation between Min Temp and Red Light Violations',

          'xaxis': {'title':'Min Temp'},

          'yaxis': {'title': 'Red Light Violations'}

}



fig = dict(data=data, layout=layout)

iplot(fig)
y0=weather[weather['winter_weather']==0]['red_light_violations']

y1=weather[weather['winter_weather']==1]['red_light_violations']



trace0 = go.Box(y=y0, name='No Winter Weather')

trace1 = go.Box(y=y1, name='Winter Weather')



data = [trace0, trace1]

fig=dict(data=data)

iplot(fig)
y0=weather[weather['snow_ice_pellets']=='0']['red_light_violations']

y1=weather[weather['snow_ice_pellets']=='1']['red_light_violations']



trace0 = go.Box(y=y0, name="No Snow/Ice")

trace1 = go.Box(y=y1, name="Snow/Ice")



data = [trace0, trace1]

fig=dict(data=data)

iplot(fig)
#Create a subset of the cam_data to only include the number of days selected

last_thirty=cam_df[(cam_df['date']>='2018-07-01')&(cam_df['date']<='2018-07-30')]



#plot the time series data

data = [go.Scatter(x=last_thirty['date'], y=last_thirty['VIOLATIONS'])]



layout = {'title': 'Number of violations between {:%x} and {:%x}'.format(last_thirty['date'].min(),last_thirty['date'].max()),

         'yaxis': {'range': [0,last_thirty['VIOLATIONS'].max()]},

         'shapes': shape(last_thirty['date'].min(),last_thirty['date'].max(),.5)}

fig = dict(data=data, layout=layout)

iplot(fig)

dow=cam_data.groupby([cam_data['VIOLATION DATE'].dt.day_name(),cam_data['VIOLATION DATE'].dt.dayofweek])[['VIOLATIONS']].agg('sum')

dow['temp']=dow.index

dow[['dow', 'day_num']]=dow.temp.apply(pd.Series)

dow=dow.reset_index(drop=True)

dow=dow.drop(['temp'], axis=1)

dow=dow.sort_values(by=['day_num'])
x=dow['dow']

y=dow['VIOLATIONS']



data=[go.Bar(x=x, y=y)]

iplot(data, filename='basic-bar')
y0=cam_df[cam_df['date'].dt.dayofweek==0]['VIOLATIONS']

y1=cam_df[cam_df['date'].dt.dayofweek==1]['VIOLATIONS']

y2=cam_df[cam_df['date'].dt.dayofweek==2]['VIOLATIONS']

y3=cam_df[cam_df['date'].dt.dayofweek==3]['VIOLATIONS']

y4=cam_df[cam_df['date'].dt.dayofweek==4]['VIOLATIONS']

y5=cam_df[cam_df['date'].dt.dayofweek==5]['VIOLATIONS']

y6=cam_df[cam_df['date'].dt.dayofweek==6]['VIOLATIONS']



trace0 = go.Box(y=y0, name="Monday")

trace1 = go.Box(y=y1, name="Tuesday")

trace2 = go.Box(y=y2, name="Wednesday")

trace3 = go.Box(y=y3, name="Thursday")

trace4 = go.Box(y=y4, name="Friday")

trace5 = go.Box(y=y5, name="Saturday")

trace6 = go.Box(y=y6, name="Sunday")



data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]

fig=dict(data=data)

iplot(fig)
#thank you