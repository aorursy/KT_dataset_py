#More examples here: https://plot.ly/python
import os

import numpy as np

import pandas as pd

import plotly as py

import plotly.graph_objs as go
N = 30     # Number of boxes



# generate an array of rainbow colors by fixing the saturation and lightness of the HSL representation of colour 

# and marching around the hue. 

# Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]



# Each box is represented by a dict that contains the data, the type, and the colour. 

# Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:

data = [{

    'y': 3.5*np.sin(np.pi * i/N) + i/N+(1.5+0.5*np.cos(np.pi*i/N))*np.random.rand(10), 

    'type':'box',

    'marker':{'color': c[i]}

    } for i in range(int(N))]



# format the layout

layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False},

          'yaxis': {'zeroline':False,'gridcolor':'white'},

          'paper_bgcolor': 'rgb(233,233,233)',

          'plot_bgcolor': 'rgb(233,233,233)',

          }



py.offline.plot(data)
trace1 = go.Scatter(

    y = np.random.randn(500),

    mode='markers',

    marker=dict(

        size=16,

        color = np.random.randn(500), #set color equal to a variable

        colorscale='Viridis',

        showscale=True

    )

)

data = [trace1]



py.offline.plot(data, filename='scatter-plot-with-colorscale.html')
dataset = pd.read_csv('../input/gapminderDataFiveYear.csv')



years = ['1952', '1962', '1967', '1972', '1977', '1982', '1987', '1992', '1997', '2002', '2007']

# make list of continents

continents = []

for continent in dataset['continent']:

    if continent not in continents:

        continents.append(continent)

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



# fill in most of layout

figure['layout']['xaxis'] = {'range': [30, 85], 'title': 'Life Expectancy'}

figure['layout']['yaxis'] = {'title': 'GDP per Capita', 'type': 'log'}

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1952',

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

year = 1952

for continent in continents:

    dataset_by_year = dataset[dataset['year'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]



    data_dict = {

        'x': list(dataset_by_year_and_cont['lifeExp']),

        'y': list(dataset_by_year_and_cont['gdpPercap']),

        'mode': 'markers',

        'text': list(dataset_by_year_and_cont['country']),

        'marker': {

            'sizemode': 'area',

            'sizeref': 200000,

            'size': list(dataset_by_year_and_cont['pop'])

        },

        'name': continent

    }

    figure['data'].append(data_dict)

    

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for continent in continents:

        dataset_by_year = dataset[dataset['year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['continent'] == continent]



        data_dict = {

            'x': list(dataset_by_year_and_cont['lifeExp']),

            'y': list(dataset_by_year_and_cont['gdpPercap']),

            'mode': 'markers',

            'text': list(dataset_by_year_and_cont['country']),

            'marker': {

                'sizemode': 'area',

                'sizeref': 200000,

                'size': list(dataset_by_year_and_cont['pop'])

            },

            'name': continent

        }

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)



    

figure['layout']['sliders'] = [sliders_dict]



py.offline.plot(figure, filename='gapminderDataFiveYear.html')