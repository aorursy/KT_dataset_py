import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import time

import warnings

warnings.filterwarnings('ignore')
terror_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1',

                          usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])

terror_data = terror_data.rename(

    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',

             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})



terror_data = terror_data.sort_values(['country'], ascending = False)
terror_data = terror_data[np.isfinite(terror_data.latitude)]

terror_data.head()
countries = list(set(terror_data.country))



country_mean_kills = []

for country in countries:

    country_mean_kills.append(terror_data.fatalities[terror_data.country == country].sum())



print('Number of people killed per attack by Country\n')

for i, country in enumerate(countries):

    print('{}:{}'.format(country, round(country_mean_kills[i],2)))
data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = country_mean_kills,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Number of\nKills')

            )

       ]



layout = dict(

    title = 'Number of people killed per attack by Country (1970 - 2015)',

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