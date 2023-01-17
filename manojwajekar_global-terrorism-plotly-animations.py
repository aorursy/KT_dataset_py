import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

from IPython.display import display, HTML

init_notebook_mode(connected=True)
file = '../input/gtd/globalterrorismdb_0617dist.csv'

gtd_main = pd.read_csv(file,encoding = "ISO-8859-1",low_memory=False)

gtd_main.info()
# Using only important columns

gtd = gtd_main[['iyear','imonth','iday','country_txt','region_txt','provstate','city','latitude','longitude','attacktype1_txt','targtype1_txt','gname','weaptype1_txt','nkill','nwound']]
gtd['text'] = gtd['city'] + '<br>' + gtd['country_txt'] + '<br>' + gtd['gname'] + '<br>' + gtd['iyear'].apply(str) + '<br>' + 'Killed:  ' + abs(gtd['nkill']).apply(str)
# for animation considering records with more than 10 victims. 

gtd = gtd[np.isfinite(gtd['nkill'])]

gtd = gtd[gtd.nkill > 10]
limits = [(0,200),(200,400),(400,1000),(1000,2000)]

colors = ["rgb(252,187,161)","rgb(251,106,74)","rgb(203,24,29)","rgb(103,0,13)","lightgrey"]

events = []

years  = [ i for i in range(1970,2015,1)]



#make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

    #'config': {'scrollzoom': True}

}



figure['layout']['title'] ='World Terrorism Dataset'

figure['layout']['showlegend'] = False

figure['layout']['geo'] = dict(resolution=50,

            projection= dict(type = 'Mercator'),

            showland = True,

            showcoastlines = False,

            landcolor = 'rgb(217, 217, 217)',

            subunitwidth=1,

            countrywidth=1,

            subunitcolor="rgb(255, 255, 255)",

            countrycolor="rgb(255, 255, 255)"

        )



figure['layout']['sliders'] = {

    'args': [

        'sliders.value', {

            'duration': 400,

            'ease': 'cubic-in-out'

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

    'transition': {'duration': 500, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



#Make data

year = 1970

for i in range(len(limits)):

    lim = limits[i]

    gtd_sub = gtd[(gtd.nkill >= lim[0]) & (gtd.nkill < lim[1])]

    #for year in years:

    gtd_sub_byyear = gtd_sub[gtd_sub.iyear == year]  

    data_dict = dict(

        type = 'scattergeo',

        lon = gtd_sub_byyear['longitude'],

        lat = gtd_sub_byyear['latitude'],

        text = gtd_sub_byyear['text'] ,

        marker = dict(

        size = gtd_sub_byyear['nkill'],

        color = colors[i],

        line = dict(width=0.5),

        sizemode = 'area'),

        name = '{0} - {1}'.format(lim[0],lim[1]) )

    figure['data'].append(data_dict)



#Make Frames

for year in years:

        frame = {'data': [], 'name': str(year)}

        for i in range(len(limits)):

            lim = limits[i]

            gtd_sub = gtd[(gtd.nkill >= lim[0]) & (gtd.nkill < lim[1])]

            gtd_sub_byyear = gtd_sub[gtd_sub.iyear == year]

            data_dict = dict(

                type = 'scattergeo',

                lon = gtd_sub_byyear['longitude'],

                lat = gtd_sub_byyear['latitude'],

                text = gtd_sub_byyear['text'] ,

                marker = dict(

                size = gtd_sub_byyear['nkill'],

                color = colors[i],

                line = dict(width=0.5),

                sizemode = 'area'),

                name = '{0} - {1}'.format(lim[0],lim[1]) )

            frame['data'].append(data_dict)

        figure['frames'].append(frame)

        slider_step = {'args': [

            [year],

            {'frame': {'duration': 500, 'redraw': False},

             'mode': 'immediate',

             'transition': {'duration': 500}}

             ],

             'label': year,

             'method': 'animate'}

        sliders_dict['steps'].append(slider_step)



figure['layout']['sliders'] = [sliders_dict]

iplot(figure,validate=False)