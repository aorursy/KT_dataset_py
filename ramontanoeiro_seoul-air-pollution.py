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
df = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")
df.head()
df['Station code'].unique()
table = df.Address.groupby(df.Address, as_index=True)

print("The dataset contains ",len(table), "different addresses")

table1 = df.Latitude.groupby(df.Latitude, as_index=True)

print("The dataset contains ",len(table1), "different Latitudes")

table2 = df.Longitude.groupby(df.Longitude, as_index=True)

print("The dataset contains ",len(table2), "different Longitudes")
df.drop("Address", axis=1, inplace=True)
df['Measurement date'] = pd.to_datetime(df['Measurement date'])
polluents = {'SO2':[0.02,0.05,0.15,1],'NO2':[0.03,0.06,0.2,2],'CO':[2,9,15,50],'O3':[0.03,0.09,0.15,0.5],'PM2.5':[15,35,75,500],'PM10':[30,80,150,600]}

quality = ['Good','Normal','Bad','Very Bad']

seoul_standard = pd.DataFrame(polluents, index=quality)

seoul_standard
df_101 = pd.DataFrame(df.loc[(df['Station code']==101)])
df_101.head()
df_101.drop("Station code", axis=1, inplace=True)
import plotly

import plotly.graph_objs as go

import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['SO2'])]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
print("We have", df_101['SO2'].loc[(df_101['SO2']<0)].count(),"negative values for SO2")
print("We have", df_101['NO2'].loc[(df_101['NO2']<0)].count(),"negative values for NO2")
print("We have", df_101['O3'].loc[(df_101['O3']<0)].count(),"negative values for O3")
print("We have", df_101['CO'].loc[(df_101['CO']<0)].count(),"negative values for CO")
print("We have", df_101['PM2.5'].loc[(df_101['PM2.5']<0)].count(),"negative values for PM2.5")
print("We have", df_101['PM10'].loc[(df_101['PM10']<0)].count(),"negative values for PM10")
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['SO2'], name='SO2'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['NO2'], name='NO2'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['CO'], name='CO'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['O3'], name='O3')]

       

##layout object

layout = go.Layout(title='Gases Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
to_drop = df_101.loc[(df_101['SO2']<0) | (df_101['NO2']<0) | (df_101['CO']<0) | (df_101['O3']<0)]

to_drop
df_101.drop(to_drop.index, axis=0, inplace=True)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['SO2'], name='SO2'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['NO2'], name='NO2'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['CO'], name='CO'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['O3'], name='O3')]

       

##layout object

layout = go.Layout(title='Gases Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM2.5'], name='PM2.5'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM10'], name='PM10'),

        ]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
to_drop_PM = df_101.loc[(df_101['PM2.5']<0) | (df_101['PM10']<0) | (df_101['PM2.5']==0) | (df_101['PM10']==0)]

to_drop_PM
df_101.drop(to_drop_PM.index, axis=0, inplace=True)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM2.5'], name='PM2.5'),

        go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM10'], name='PM10'),

        ]

       

##layout object

layout = go.Layout(title='SO2 Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



## Plotting

py.iplot(fig)
df_101.head(2)
df_101.tail(2)
seoul_standard
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['SO2'])]

       

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
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['NO2'])]

       

##layout object

layout = go.Layout(title='NO2 Levels',

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

            y0=0.03,

            x1='2019-12-31 23:00:00',

            y1=0.03,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=0.06,

            x1='2019-12-31 23:00:00',

            y1=0.06,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['CO'])]

       

##layout object

layout = go.Layout(title='CO Levels',

                    yaxis={'title':'Level (ppm)'},

                    xaxis={'title':'Date'})

                    

## Figure object



fig = go.Figure(data=data, layout=layout)



    



##Adding the text and positioning it

fig.add_trace(go.Scatter(

    x=['2017-03-01 00:00:00', '2017-07-31 23:00:00'],

    y=[15, 10],

    text=["Safe Level - Green", "Normal Level - Orange"],

    mode="text",

            ))



##Adding horizontal line

fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=2,

            x1='2019-12-31 23:00:00',

            y1=2,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=9,

            x1='2019-12-31 23:00:00',

            y1=9,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['O3'])]

       

##layout object

layout = go.Layout(title='O3 Levels',

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

            y0=0.03,

            x1='2019-12-31 23:00:00',

            y1=0.03,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=0.09,

            x1='2019-12-31 23:00:00',

            y1=0.09,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM2.5'])]

       

##layout object

layout = go.Layout(title='PM2.5 Levels',

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

            y0=15,

            x1='2019-12-31 23:00:00',

            y1=15,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=35,

            x1='2019-12-31 23:00:00',

            y1=35,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)
data = [go.Scatter(x=df_101['Measurement date'],

                   y=df_101['PM10'])]

       

##layout object

layout = go.Layout(title='PM10 Levels',

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

            y0=30,

            x1='2019-12-31 23:00:00',

            y1=30,

            line=dict(

                color="Green",

                width=4,

                dash="dashdot",

            ))



fig.add_shape(

        # Line Horizontal

            type="line",

            x0='2017-01-01 00:00:00',

            y0=80,

            x1='2019-12-31 23:00:00',

            y1=80,

            line=dict(

                color="Orange",

                width=4,

                dash="dashdot",

            ))





## Plotting

py.iplot(fig)