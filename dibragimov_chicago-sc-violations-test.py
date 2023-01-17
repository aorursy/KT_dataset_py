# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime

# For data files available in the "../input/" directory.

import os

#print(os.listdir("../input/red-light-camera-violations.csv"))

# Any results you write to the current directory are saved as output.



import plotly

#import chart_studio

#from chart_studio.plotly import iplot

#import chart_studio.plotly as py

#import plotly.plotly as py

#import plotly.graph_objs as go

import plotly.graph_objects as go

# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



#show graphs inline for simple charts

%matplotlib inline
#import data

red_light_viol = pd.read_csv("../input/red-light-camera-violations.csv", index_col=0)

camera_loc = pd.read_csv("../input/red-light-camera-locations.csv", index_col=0)



#fix columns to contain the same data for further join (one contains AND while second has dash, one is uppercase while seond is capitalized)

camera_loc['intersect'] = camera_loc.index.str.upper()

camera_loc['intersect'] = camera_loc['intersect'].str.replace('-', ' AND ')

red_light_viol['intersect'] = red_light_viol.index.str.upper()



#merge (join) datasets

merged_rlviol = pd.merge(red_light_viol, camera_loc, on = "intersect", how='inner')



#remove and rename columns

merged_rlviol = merged_rlviol.rename(columns={'LATITUDE_y':'LATITUDE','LONGITUDE_y':'LONGITUDE', 'LOCATION_y':'LOCATION'})

merged_rlviol = merged_rlviol.drop(columns=['X COORDINATE', 'Y COORDINATE', 'LATITUDE_x', 'LONGITUDE_x', 'LOCATION_x', 'FIRST APPROACH', 'SECOND APPROACH', 'THIRD APPROACH'])

#total violations

print('Total number of Red Light Violations: ' + str(red_light_viol['VIOLATIONS'].sum()))
red_light_viol['viol_datetime'] = pd.to_datetime(red_light_viol['VIOLATION DATE'])

df_rlviol = red_light_viol.groupby([red_light_viol['viol_datetime'].dt.year, red_light_viol['viol_datetime'].dt.month])['VIOLATIONS'].agg('sum')

#convert to DataFrame

df_rlviol = df_rlviol.to_frame()

df_rlviol['date'] = df_rlviol.index

# rename column

df_rlviol = df_rlviol.rename(columns={df_rlviol.columns[0]:"violations"})

# re-parse dates

df_rlviol['date'] = pd.to_datetime(df_rlviol['date'], format="(%Y, %m)")

# remove index

df_rlviol = df_rlviol.reset_index(drop=True)

# get month and year of meet

df_rlviol['year'] = df_rlviol.date.dt.year

df_rlviol['month'] = df_rlviol.date.dt.month

#df_rlviol.plot.line(x='viol_datetime')  #### draw a line chart
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x=df_rlviol.date, y=df_rlviol.violations)]



# specify the layout of our figure

layout = dict(title = "Number of Red Light Violations per Month",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
mapbox_access_token ='pk.eyJ1IjoiZGlicmFnaW1vdiIsImEiOiJjanozemgzMGMwODloM2ltbGt5czc2ejRwIn0.ulQDw9pMGRlPmdLNafyJcw'
#calculate violatipns per camera to show which camera catches how many violations (total)

agg_viols = merged_rlviol.groupby([merged_rlviol['CAMERA ID'], merged_rlviol['LATITUDE'], merged_rlviol['LONGITUDE']], as_index=False)['VIOLATIONS'].agg('sum')



# Plot with plotly and Mapbox

mapbox_access_toke = 'pk.eyJ1Ijoid3RzY290dCIsImEiOiJjanB2ZXJ0bXEwMWt3M3h0ZGExbXlqNDFlIn0.YDNTY8VZk0Ytm-Q0J8Zp_Q'



data = [

    go.Scattermapbox(

        lat=agg_viols.LATITUDE,

        lon=agg_viols.LONGITUDE,

        mode='markers',

        text=agg_viols.VIOLATIONS,

        marker=dict(

            size=7,

            color=agg_viols.VIOLATIONS,

            colorscale='Electric',

            colorbar=dict(

                title='VIOLATIONS'

            )

        ),

    )

]



layout = go.Layout(

    autosize=True,

    hovermode='closest',

    title='Chicago Red Light Violations, Updated ' ,#+ str(today.date()

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