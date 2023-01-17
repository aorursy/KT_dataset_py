import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')



global_temp_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



global_temp_country_clear = global_temp_country_clear.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Let's average temperature for each country



countries = np.unique(global_temp_country_clear['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == 

                                               country]['AverageTemperature'].mean())





    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')

print("It is possible to notice that Russia has one of the lowest average temperature (like a Canada). The lowest temperature in Greenland (it is distinctly visible on the map). The hottest country in Africa, on the equator. It is quite natural.")
global_temp = pd.read_csv("../input/GlobalTemperatures.csv")



#Extract the year from a date

years = np.unique(global_temp['dt'].apply(lambda x: x[:4]))

mean_temp_world = []

mean_temp_world_uncertainty = []



for year in years:

    mean_temp_world.append(global_temp[global_temp['dt'].apply(

        lambda x: x[:4]) == year]['LandAverageTemperature'].mean())

    mean_temp_world_uncertainty.append(global_temp[global_temp['dt'].apply(

                lambda x: x[:4]) == year]['LandAverageTemperatureUncertainty'].mean())



trace0 = go.Scatter(

    x = years, 

    y = np.array(mean_temp_world) + np.array(mean_temp_world_uncertainty),

    fill= None,

    mode='lines',

    name='Uncertainty top',

    line=dict(

        color='rgb(0, 255, 255)',

    )

)

trace1 = go.Scatter(

    x = years, 

    y = np.array(mean_temp_world) - np.array(mean_temp_world_uncertainty),

    fill='tonexty',

    mode='lines',

    name='Uncertainty bot',

    line=dict(

        color='rgb(0, 255, 255)',

    )

)



trace2 = go.Scatter(

    x = years, 

    y = mean_temp_world,

    name='Average Temperature',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)

data = [trace0, trace1, trace2]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Average Temperature, °C'),

    title='Average land temperature in world',

    showlegend = False)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)