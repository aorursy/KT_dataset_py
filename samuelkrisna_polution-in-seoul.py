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
df_polution = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")

df_polution.head()
df_polution['Station code'].unique()

address = df_polution.Address.groupby(df_polution.Address, as_index=True)

print("The dataset contains ",len(address), "different addresses")

latitude = df_polution.Latitude.groupby(df_polution.Latitude, as_index=True)

print("The dataset contains ",len(latitude), "different Latitudes")

longitude = df_polution.Longitude.groupby(df_polution.Longitude, as_index=True)

print("The dataset contains ",len(longitude), "different Longitudes")
df_polution['Measurement date'] = pd.to_datetime(df_polution['Measurement date'])

polluents = {'SO2':[0.02,0.05,0.15,1],'NO2':[0.03,0.06,0.2,2],'CO':[2,9,15,50],'O3':[0.03,0.09,0.15,0.5],'PM2.5':[15,35,75,500],'PM10':[30,80,150,600]}

quality = ['Good','Normal','Bad','Very Bad']

seoul_standard = pd.DataFrame(polluents, index=quality)

seoul_standard
import plotly

import plotly.graph_objs as go

import plotly.offline as py

df_visualisasi = pd.DataFrame(df_polution.loc[(df_polution['Station code']==101)])

df_visualisasi.head()
plotly.offline.init_notebook_mode(connected=True)

data = [go.Scatter(x=df_visualisasi['Measurement date'],

                   y=df_visualisasi['SO2'])]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_visualisasi['Measurement date'],

                   y=df_visualisasi['PM2.5'], name='PM2.5'),

        go.Scatter(x=df_visualisasi['Measurement date'],

                   y=df_visualisasi['PM10'], name='PM10'),

        ]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_visualisasi['Measurement date'],

                   y=df_visualisasi['SO2'])]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



    



##Adding the text and positioning it

fig.add_trace(go.Scatter(

    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],

    y=[0.2, 0.15],

    text=["Safe Level - Green", "Normal Level - Orange"],

    mode="text",

            ))



##Adding horizontal line

fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=0.02,

            x1='2019-12-31 23:00:00',

            y1=0.02,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=0.05,

            x1='2019-12-31 23:00:00',

            y1=0.05,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)