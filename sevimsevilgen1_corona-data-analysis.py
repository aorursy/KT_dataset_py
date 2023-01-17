import numpy as np # Linear Algebre

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from plotly.offline import init_notebook_mode, plot ,iplot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go # plotly graphical object



# Input data files are available in the "../input/" directory.

import warnings 

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
data = pd.read_csv('/kaggle/input/datasets_519149_986288_novel_corona_cleaned_latest.csv')

data
data["Month"]= [int(each.split("-")[1]) for each in data.iloc[:,4]]



dataset = data.loc[:,["Country/Region","Lat","Long","Last Update","Deaths","Month"]]

dataset

dataset = dataset[dataset['Deaths']>0]

dataset.head(60)
months = [str(each) for each in list(dataset.Month.unique())] #oluşturduğumuz kolonu unique yapıyoruz.

deaths = dataset.Deaths.unique()

ulkeler = ['Australia','China','France','Italy','US','Spain'] #Hastalık Tipimizi belirliyoruz.

custom_colors = {

    'Australia':'rgb(140,63,12)',

    'China': 'rgb(16,150,32)',

    'France':'rgb(56,47,220)',

    'Italy':'rgb(170,21,63)',

    'US':'rgb(46,78,120)',

    'Spain':'rgb(110,154,212)'

    

}

# make figure

figure = {

    'data': [], #coğrafik harita

    'layout': {}, #Başlıkların bulunduğu kısım

    'frames': [] #ekliyeceğimiz butanlar olacak.

}



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = { #Kaydırıcı

    'args': [

        'transition', {

            'duration': 400, #Süresi

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1',

    'plotlycommand': 'animate',

    'values': months,

    'visible': True

}

figure['layout']['updatemenus'] = [ #butonlar

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            { #ikinci buton

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left', # sol kısımda gözükecek

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

        'prefix': 'Month:',

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



# make data

Month = 1

for ty in ulkeler:

    dataset_by_month = dataset[dataset['Month'] == Month]

    dataset_by_month_and_cont = dataset_by_month[dataset_by_month['Country/Region'] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = dataset['Long'],

    lat = dataset['Lat'],

    hoverinfo = 'text',

    text = ty,

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = custom_colors[ty],

        opacity = 0.7),

)

    figure['data'].append(data_dict)

    

# make frames



for Month in months:

    frame = {'data': [], 'name': str(Month)}

    

    for ty in ulkeler:

        dataset_by_month = dataset[dataset['Month'] == int(Month)]

        dataset_by_month_and_cont = dataset_by_month[dataset_by_month['Country/Region'] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_month_and_cont['Long'],

                lat = dataset_by_month_and_cont['Lat'],

                hoverinfo = 'text',

                text = ty,

                mode = 'markers',

                marker=dict(

                    sizemode = 'area',

                    sizeref = 1,

                    size= 10 ,

                    line = dict(width=1,color = "white"),

                    color = custom_colors[ty],

                    opacity = 0.7),

                name = ty

            )

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [Month],

        {'frame': {'duration': 700, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 700}}

     ],

     'label': Month,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Corona"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)








