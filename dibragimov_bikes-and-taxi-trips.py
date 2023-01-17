# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.graph_objs as go

import geopandas as gpd

from shapely.geometry import Point, Polygon, MultiPolygon

import matplotlib

import shapely.wkt



#show graphs inline for simple charts

%matplotlib inline



# code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
comm_df = pd.read_csv('/kaggle/input/Comm_Areas_1.csv', sep=';', header=0, encoding='utf-8', decimal='.' )

bike_df = pd.read_csv('/kaggle/input/divvy_bikes_trips_agg.csv', sep=',', header=0, encoding='utf-8', decimal='.' )

merged_bike_df = bike_df.merge(comm_df, how='inner', left_on='Community Areas', right_on='CommArea')

geo_bike_df = gpd.GeoDataFrame(merged_bike_df, geometry=merged_bike_df['Geometry'].apply(shapely.wkt.loads))

taxi_df = pd.read_csv('/kaggle/input/taxi_trips_agg.csv', sep=',', header=0, encoding='utf-8', decimal='.' )

merged_taxi_df = taxi_df.merge(comm_df, how='inner', left_on='pickup_community_area', right_on='CommArea')

geo_taxi_df = gpd.GeoDataFrame(merged_taxi_df, geometry=merged_taxi_df['Geometry'].apply(shapely.wkt.loads))

clmn = 'distance'

vmin_bike, vmax_bike = geo_bike_df[clmn].min()*0.9, geo_bike_df[clmn].max()*1.1

import matplotlib.pyplot as plt

#fig, ax = plt.subplots(1, figsize=(10, 6))

fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(20, 6))

geo_bike_df.plot(column=clmn, cmap='Blues', linewidth=0.8, ax=axes[0], edgecolor='0.8') ## numberOfTrips ## duration ## distance

# remove the axis

axes[0].axis('off')



# add a title

axes[0].set_title('Average Distance for Bike Rentals\n by Community Area', \

              fontdict={'fontsize': '25',

                        'fontweight' : '3'})

# Create colorbar as a legend

fig.colorbar(axes[0].collections[0], ax=axes[0])



geo_taxi_df.plot(column=clmn, cmap='Oranges', linewidth=0.8, ax=axes[1], edgecolor='0.8') ## duration ## av_length ## numOfTrips

# remove the axis

axes[1].axis('off')



# add a title

axes[1].set_title('Average Distance for Taxi Trips\n by Community Area', \

              fontdict={'fontsize': '25',

                        'fontweight' : '3'})

# Create colorbar as a legend

fig.colorbar(axes[1].collections[0], ax=axes[1])
#### load aggregated bikes data

bikes_df = pd.read_csv('/kaggle/input/divvy_bikes_trips.csv', sep=',', header=0, encoding='utf-8', decimal='.' )

#aggregate by month and year

bikes_df = bikes_df.groupby([bikes_df['start_year'], bikes_df['start_month']])['distance'].agg('mean')

#convert to DataFrame

bikes_df = bikes_df.to_frame()

bikes_df['date'] = bikes_df.index

# re-parse dates

bikes_df['date'] = pd.to_datetime(bikes_df['date'], format="(%Y, %m)")

# remove index

bikes_df = bikes_df.reset_index(drop=True)

#### load aggregated taxi data

taxis_df = pd.read_csv('/kaggle/input/bq_results_chicago_taxi.csv', sep=',', header=0, encoding='utf-8', decimal='.' )

taxis_df = taxis_df[taxis_df['trip_start_year'] < 2019] #### get it for 2018 only - to be on par with Divvy

#aggregate by month and year

taxis_df = taxis_df.groupby([taxis_df['trip_start_year'], taxis_df['trip_start_month']])['length'].agg('mean')

#convert to DataFrame

taxis_df = taxis_df.to_frame()

taxis_df['date'] = taxis_df.index

# re-parse dates

taxis_df['date'] = pd.to_datetime(taxis_df['date'], format="(%Y, %m)")

# remove index

taxis_df = taxis_df.reset_index(drop=True)
#### create two lines and show it in interactive mode

trace1 = go.Scatter(

                x=bikes_df['date'], y=bikes_df['distance'],

                mode = "lines+markers",

                name = "Average Distance on Bike per Month",

                marker = dict(color = 'rgb(102,102,255)')#, text = df_groupby_datebr['ZHVI_1bedroom']

)

trace2 = go.Scatter(

                x=taxis_df['date'], y=taxis_df['length'],

                mode = "lines+markers",

                name = "Average Distance on Taxi per Month",

                marker = dict(color = 'rgb(178, 102, 255)')#, text = df_groupby_datebr['ZHVI_2bedroom'] ##rgb(102,178,255)

)

#### display 2 lines in a graph

data = [trace1, trace2]



# specify the layout of our figure

layout = dict(title = "Average Distance per Month in 2018",

              xaxis= dict(title= 'Dates',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
agg_bikes = pd.read_csv('/kaggle/input/divvy_agg_count.csv', sep=',', header=0, encoding='utf-8', decimal='.')



# Plot with plotly and Mapbox

mapbox_access_token ='pk.eyJ1IjoiZGlicmFnaW1vdiIsImEiOiJjanozemgzMGMwODloM2ltbGt5czc2ejRwIn0.ulQDw9pMGRlPmdLNafyJcw'



data = [

    go.Scattermapbox(

        lat=agg_bikes['FROM LATITUDE'],

        lon=agg_bikes['FROM LONGITUDE'],

        mode='markers',

        text=agg_bikes['Descr'],

        marker=dict(

            size=7,

            color=agg_bikes['NumOfTrips'],

            colorscale='Electric',

            colorbar=dict(

                title='Trips'

            )

        ),

    )

]



layout = go.Layout(

    autosize=True,

    hovermode='closest',

    title='Chicago Divvy Bikes Statistics, Updated ' ,#+ str(today.date()

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=41.881,

            lon=-87.666

        ),

        pitch=0,

        zoom=10

    ),

)



fig = dict(data=data, layout=layout)

iplot(fig, filename='Chicago Mapbox')