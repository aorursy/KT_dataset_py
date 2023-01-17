import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

language_data = pd.read_csv('../input/data.csv', usecols=[0, 1, 5, 7, 10, 12, 13])
language_data = language_data.rename(
    columns={'Name in English':'language', 'Country codes alpha 3':'locations',
             'Degree of endangerment':'risk', 'Number of speakers':'population'})
language_data.columns = language_data.columns.str.lower()
language_data['risk'] = language_data['risk'].str.title()
language_data['population'] = language_data['population'].fillna(-1)

# endangered or extinct languages in United States only (224 rows)
language_usa = language_data[language_data['locations'].str.contains('USA') == True]
mask = language_usa['language'].isin(['Kwak\'wala','Okanagan','Central Alaskan Yupik (2)'])
language_usa = language_usa[~mask]
language_usa['language'] = language_usa['language'].str.replace(
                                                    ' \(United States of America\)', '')
language_usa['risk'] = language_usa['risk'].replace(
    ['Vulnerable', 'Definitely Endangered', 'Severely Endangered',
     'Critically Endangered', 'Extinct'], [1, 2, 3, 4, 5])
language_usa = language_usa[['language', 'risk', 'population', 'latitude', 'longitude']]

# missing estimates for population replaced with median value
risk_levels = [2, 3, 4, 5]
for i in risk_levels:
    language_usa.loc[(language_usa.risk == i) & (language_usa.population < 0), 'population'
                    ] = language_usa['population'][language_usa.risk == i].median()
language_usa.loc[(language_usa.risk == 5), 'population'] = 0
language_usa['population'] = language_usa['population'].astype(int)
labels = ['Isolated', 'Threatened', 'Endangered', 'Abandoned', 'Extinct']
colors = ['rgb(0, 157, 220)', 'rgb(128, 206, 237)', 'rgb(255, 182, 128)',
          'rgb(255, 115, 13)', 'rgb(242, 23, 13)']

traces = []
for i in range(1, 6):
    traces.append(dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = language_usa[language_usa.risk == i]['longitude'],
        lat = language_usa[language_usa.risk == i]['latitude'],
        text = language_usa[language_usa.risk == i]['language'],
        mode = 'markers',
        name = labels[i-1],
        marker = dict( 
            size = 12,
            opacity = 0.85,
            color = colors[i-1],
            line = dict(color = 'rgb(255, 255, 255)', width = 1)
        )
    ))

layout = dict(
         title = 'Languages by Latitude/Longitude in United States (2016)<br>'
                 '<sub>Click Legend to Display or Hide Categories</sub>',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = traces, layout = layout)
iplot(figure)
# endangered or extinct languages by population of native speakers
language_usa = language_usa.sort_values('population', ascending = False)
language_usa['text'] = language_usa['language'] + '<br>' + 'Population ' + language_usa[
                                                                 'population'].astype(str)

traces = []
for i in range(1, 6):
    traces.append(dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = language_usa[language_usa.risk == i]['longitude'],
        lat = language_usa[language_usa.risk == i]['latitude'],
        text = language_usa[language_usa.risk == i]['text'],
        mode = 'markers',
        name = labels[i-1],
        hoverinfo = 'text+name',
        marker = dict( 
            size = (language_usa[language_usa.risk == i]['population'] + 1) ** 0.18 * 6,
            opacity = 0.85,
            color = colors[i-1],
            line = dict(color = 'rgb(255, 255, 255)', width = 1)
        )
    ))

layout = dict(
         title = 'Languages by Population in United States (2016)<br>'
                 '<sub>Click Legend to Display or Hide Categories</sub>',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = traces, layout = layout)
iplot(figure)