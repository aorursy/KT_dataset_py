# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
import plotly.plotly as py # visualization library
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import warnings            
warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.
plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/scrubbed.csv")
data = data.rename(columns = {'longitude ':'longitude' })
data.info()
data.head(10)
data.tail(10)
data["country"].fillna("missing" ,inplace = True)
data.columns
data["shape"].fillna("empty" ,inplace = True)
data.country.value_counts()
data['color'] = colors = ["" for x in data.country]
data["year"]= [int(each.split()[0].split('/')[2]) for each in data.iloc[:, 0]]
mapPlotData = data[data.year > 2010]
mapPlotData = mapPlotData[mapPlotData.country != "missing"]
mapPlotData.head()
mapPlotData.year.value_counts()
mapPlotData.color[mapPlotData.country == "us"] = "rgb(0, 116, 217)"
mapPlotData.color[mapPlotData.country == "gb"] = "rgb(255, 65, 54)"
mapPlotData.color[mapPlotData.country == "ca"] = "rgb(133, 20, 75)"
mapPlotData.color[mapPlotData.country == "au"] = "rgb(255, 133, 27)"
mapPlotData.color[mapPlotData.country == "de"] = "rgb(255, 7, 4)"
#mapPlotData.color[mapPlotData.country == "missing"] = "rgb(255, 255, 255)"

mapData = [dict(
    type = 'scattergeo',
    lon = mapPlotData.longitude,
    lat = mapPlotData.latitude,
    hoverinfo = 'text',
    text = "Sigth Location: " + mapPlotData.country,
    mode = 'markers',
    marker = dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width = 1, color = "white"),
        color = mapPlotData["color"],
        opacity = 0.7),
)]
layout = dict(
    title = 'UFO Sightings Between 2011 - 2014',
    hovermode = 'closest', 
    width = 1500, 
    height = 900,
    geo = dict(showframe = False, 
               showland = True, 
               showcoastlines = True, 
               showcountries = True, 
               countrywidth = 1, 
               projection = dict(type = 'mercator'),
               landcolor = 'rgb(217, 217, 217)',
               subunitwidth = 1,
               showlakes = True,
               lakecolor = 'rgb(255, 255, 255)',
               countrycolor = "rgb(5, 5, 5)")
)

fig = go.Figure(data = mapData, layout = layout)
iplot(fig)
data["shape"].unique()
data["shape"].value_counts()
data = data[(data["shape"] == "light") | (data["shape"] == "triangle") | (data["shape"] == "circle") | (data["shape"] == "fireball") | (data["shape"] == "other")]
dataset = data.loc[:,["datetime", "country", "latitude", "longitude", "shape", "year"]]
dataset.head(10)
dataset["shape"].value_counts()
years = [str(each) for each in list(data.year.unique())]

# list of most common shapes
shapes = ['light', 'triangle', 'circle', 'fireball', 'other']

custom_colors = {
    'light': 'rgb(189, 2, 21)',
    'triangle': 'rgb(52, 7, 250)',
    'circle': 'rgb(99, 110, 250)',
    'fireball': 'rgb(0, 0, 0)',
    'other' : 'rgb(255, 255, 0)'
}

# make figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

figure['layout']['geo'] = dict(showframe = False, 
                               showland = True, 
                               showcoastlines = True, 
                               showcountries = True, 
                               countrywidth = 1, 
                               landcolor = 'rgb(217, 217, 217)',
                               subunitwidth = 1,
                               showlakes = True,
                               lakecolor = 'rgb(255, 255, 255)',
                               countrycolor = "rgb(5, 5, 5)")
figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': '1950',
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}

figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
year = 1950
for shp in shapes:
    dataset_by_year = dataset[dataset['year'] == year]
    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['shape'] == shp]    
    data_dict = dict(
        type = 'scattergeo',
        lon = dataset['longitude'],
        lat = dataset['latitude'],
        hoverinfo = 'text',
        text = shp,
        mode = 'markers',
        marker = dict(
            sizemode = 'area',
            sizeref = 1,
            size = 10 ,
            line = dict(width = 1, color = "white"),
            color = custom_colors[shp],
            opacity = 0.7),
)
    figure['data'].append(data_dict)
    
# make frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    for shp in  shapes:
        dataset_by_year = dataset[dataset['year'] == int(year)]
        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['shape'] == shp]
        data_dict = dict(
            type = 'scattergeo', 
            lon = dataset_by_year_and_cont['longitude'],
            lat = dataset_by_year_and_cont['latitude'],
            hoverinfo = 'text',
            text = shp,
            mode = 'markers',
            marker = dict(
                sizemode = 'area',
                sizeref = 1,
                size= 10 ,
                line = dict(width = 1,color = "white"),
                color = custom_colors[shp],
                opacity = 0.7),
                name = shp
        )
        frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [[year], {'frame': {'duration': 300, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 300}}], 
                   'label': year, 
                   'method': 'animate'}
    sliders_dict['steps'].append(slider_step)

figure["layout"]["autosize"] = True
figure["layout"]["title"] = "UFO Sightings - Shapes"       
figure['layout']['sliders'] = [sliders_dict]
iplot(figure)    