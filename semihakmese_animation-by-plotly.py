import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library



from plotly.offline import init_notebook_mode, plot, iplot

import plotly as py

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings            

warnings.filterwarnings("ignore") 

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/earthquake-database/database.csv")

data=data.drop([3378,7512,20650]) #bu veriler hatalı old.icin cıkardık

data["year"]=[int(each.split("/")[2]) for each in data.iloc[:,0]] #Burada sadece yıl verisine ulaşmak için yaptık 0.veri gün,1.veri ay,2.veri yıl

#ve year diye bir column olusturduk ve dataya ekledik
data.head()
data.info() #Görüldüğü üzere year adında bir int64 formatında verim oldu.
data.columns
data.Type.unique() 

#Burada yasanan felaketleri görüyoruz
dataset=data.loc[:,["Date","Latitude","Longitude","Type","Depth","Magnitude","year"]]

dataset.head()

#Latitude enlem , Longitude Boylam anlamına gelir 
years =[str(each) for each in list(data.year.unique())] #Burada datadaki yılları 1 er kez almıs olduk yani unique leri almıs olduk

types=["Earthquake","Nuclear Explosion","Explosion","Rock Burst"]

custom_colors= {

    'Earthquake': 'rgb(189, 2, 21)',

    'Nuclear Explosion': 'rgb(52, 7, 250)',

    'Explosion': 'rgb(99, 110, 250)',

    'Rock Burst': 'rgb(0, 0, 0)'   

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}

#Aynı map gibi yapıyoruz

figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = { #Slider süreyi belirler

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1965', #Baslangıc yılı

    'plotlycommand': 'animate', #Animasyon

    'values': years,

    'visible': True

}

figure['layout']['updatemenus'] = [ #Etkileşimli menu actık 

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

year = 1695

for ty in types:

    dataset_by_year = dataset[dataset['year'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = dataset['Longitude'],

    lat = dataset['Latitude'],

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

for year in years:

    frame = {'data': [], 'name': str(year)}

    for ty in types:

        dataset_by_year = dataset[dataset['year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_year_and_cont['Longitude'],

                lat = dataset_by_year_and_cont['Latitude'],

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

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Earthquake"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)