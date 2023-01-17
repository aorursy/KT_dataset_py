# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library

import matplotlib.pyplot as plt # visualization library

import plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
aerial = pd.read_csv("../input/world-war-ii/operations.csv")
# drop countries that are NaN

aerial = aerial[pd.isna(aerial.Country)==False]

# drop if target longitude is NaN

aerial = aerial[pd.isna(aerial['Target Longitude'])==False]

# Drop if takeoff longitude is NaN

aerial = aerial[pd.isna(aerial['Takeoff Longitude'])==False]

# drop unused features

drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',

             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',

             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',

             'High Explosives', 'High Explosives Type','Mission Type',

             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',

             'Incendiary Devices', 'Incendiary Devices Type',

             'Incendiary Devices Weight (Pounds)',

             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',

             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',

             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',

             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']

aerial.drop(drop_list, axis=1,inplace = True)

aerial = aerial[ aerial.iloc[:,8]!="4248"] # drop this takeoff latitude 

aerial = aerial[ aerial.iloc[:,9]!=1355]   # drop this takeoff longitude
print(aerial['Country'].value_counts())

plt.figure(figsize=(22,10))

sns.countplot(aerial['Country'])

plt.show()
print(aerial['Target Country'].value_counts()[:10])

plt.figure(figsize=(22,10))

sns.countplot(aerial['Target Country'])

plt.xticks(rotation=90)

plt.show()
# Aircraft Series

data = aerial['Aircraft Series'].value_counts()

print(data[:10])

data = [go.Bar(

            x=data[:10].index,

            y=data[:10].values,

            hoverinfo = 'text',

            marker = dict(color = 'rgba(125, 14, 22, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

    )]



layout = dict(

    title = 'Aircraft Series',

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# ATTACK

aerial["color"] = ""

aerial.color[aerial.Country == "USA"] = "rgb(0,116,217)"

aerial.color[aerial.Country == "GREAT BRITAIN"] = "rgb(255,65,54)"

aerial.color[aerial.Country == "NEW ZEALAND"] = "rgb(133,20,75)"

aerial.color[aerial.Country == "SOUTH AFRICA"] = "rgb(255,133,27)"



data = [dict(

    type='scattergeo',

    lon = aerial['Takeoff Longitude'],

    lat = aerial['Takeoff Latitude'],

    hoverinfo = 'text',

    text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = aerial["color"],

        opacity = 0.7),

)]

layout = dict(

    title = 'Countries Take Off Bases ',

    hovermode='closest',

    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, projection=dict(type='miller'),

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
airports = [ dict(

        type = 'scattergeo',

        lon = aerial['Takeoff Longitude'],

        lat = aerial['Takeoff Latitude'],

        hoverinfo = 'text',

        text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],

        mode = 'markers',

        marker = dict( 

            size=5, 

            color = aerial["color"],

            line = dict(

                width=1,

                color = "white"

            )

        ))]



targets = [ dict(

        type = 'scattergeo',

        lon = aerial['Target Longitude'],

        lat = aerial['Target Latitude'],

        hoverinfo = 'text',

        text = "Target Country: "+aerial["Target Country"]+" Target City: "+aerial["Target City"],

        mode = 'markers',

        marker = dict( 

            size=1, 

            color = "red",

            line = dict(

                width=0.5,

                color = "red"

            )

        ))]



flight_paths = []

for i in range( len( aerial['Target Longitude'] ) ):

    flight_paths.append(

        dict(

            type = 'scattergeo',

            lon = [ aerial.iloc[i,9], aerial.iloc[i,16] ],

            lat = [ aerial.iloc[i,8], aerial.iloc[i,15] ],

            mode = 'lines',

            line = dict(

                width = 0.7,

                color = 'black',

            ),

            opacity = 0.7,

        )

    )

    

layout = dict(

    title = 'Bombing Paths from Attacker Country to Target ',

    hovermode='closest',

    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, projection=dict(type='miller'),

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

)

    

fig = dict( data=flight_paths + airports+targets, layout=layout )

iplot( fig )
data = pd.read_csv("../input/earthquake-database/database.csv")

data = data.drop([3378,7512,20650])

data["year"]= [int(each.split("/")[2]) for each in data.iloc[:,0]]
data.head()
data.Type.unique()
dataset = data.loc[:,["Date","Latitude","Longitude","Type","Depth","Magnitude","year"]]

dataset.head()
years = [str(each) for each in list(data.year.unique())]  # str unique years

# make list of types

types = ['Earthquake', 'Nuclear Explosion', 'Explosion', 'Rock Burst']

custom_colors = {

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



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1965',

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