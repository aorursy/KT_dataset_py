import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/GHG-Mexico-1990-2010.csv")

GHG_list = list(df['GHG'].unique())

GHG_list.sort() 

  

print(GHG_list)
Sector_list = list(df['Sector'].unique())

Sector_list.sort() 

  

print(Sector_list)
Subsector_list = list(df['Subsector'].unique())

Subsector_list.sort() 

  

print(Subsector_list)
Year_list = list(df['Year'].unique())

Year_list.sort() 

  

print(Year_list)




fig = {

    'data': [

        {

            'labels': GHG_list,

            'values': [2664105, 9490380, 1138450],

            'type': 'pie',

            'name': 'GHG',

            'marker': {'colors': ['rgb(56, 75, 126)',

                                  'rgb(18, 36, 37)',

                                  'rgb(34, 53, 101)',

                                  'rgb(36, 55, 57)',

                                  'rgb(6, 4, 4)']},

            'domain': {'x': [0, .48],

                       'y': [0, .49]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        },

        {

            'labels':  Year_list,

            'values': [764540, 587591, 577542, 586838, 560593, 577511, 591317,604083, 580771, 614670, 633377, 538402, 610093, 623283, 656220, 658342, 682343, 707889, 733920, 676235, 727364                                 ],

             

            'marker': {'colors': ['rgb(177, 127, 38)',

                                  'rgb(205, 152, 36)',

                                  'rgb(99, 79, 37)',

                                  'rgb(129, 180, 179)',

                                  'rgb(124, 103, 37)']},

            'type': 'pie',

            'name': 'Year',

            'domain': {'x': [.52, 1],

                       'y': [0, .49]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'



        },

        {

            'labels': Sector_list,

            'values': [1735849, 8625317, 790810, 1517720, 623237],

            'marker': {'colors': ['rgb(33, 75, 99)',

                                  'rgb(79, 129, 102)',

                                  'rgb(151, 179, 100)',

                                  'rgb(175, 49, 35)',

                                  'rgb(36, 73, 147)']},

            'type': 'pie',

            'name': 'Sector',

            'domain': {'x': [0, .48],

                       'y': [.51, 1]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        },

        {

            'labels':   Subsector_list,

            'values':  [7377,52799,264153,772236,982,1263662,18501,2931990,1185809,152567,126237,611773,692118,4178,1517719,805884,2551735,333205,],

           

            'marker': {'colors': ['rgb(146, 123, 21)',

                                  'rgb(177, 180, 34)',

                                  'rgb(206, 206, 40)',

                                  'rgb(175, 51, 21)',

                                  'rgb(35, 36, 21)']},

            'type': 'pie',

            'name':'Subsector',

            'domain': {'x': [.52, 1],

                       'y': [.51, 1]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'

        }

        

    ],

    'layout': {'title': '',

               'showlegend': False}

}



iplot(fig)
import plotly.plotly as py

import plotly.graph_objs as go

trace0 = go.Bar(

    x=Year_list,

    



    y=  [764540, 587591, 577542, 586838, 560593, 577511, 591317,604083, 580771, 614670, 633377, 538402, 610093, 623283, 656220, 658342, 682343, 707889, 733920, 676235, 727364                                 ],

    text=['', '', ''],

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        )

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Yearly',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='text-hover-bar')
import plotly.plotly as py

import plotly.graph_objs as go

trace0 = go.Bar(

    x=GHG_list,

    y= [2664105, 9490380, 1138450],

    text=['', '', ''],

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        )

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Total Green House Gasses Emission',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='text-hover-bar')
import plotly.plotly as py

import plotly.graph_objs as go

trace0 = go.Bar(

    x=Sector_list,

    y= [1735849, 8625317, 790810, 1517720, 623237],

    text=['', '', ''],

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        )

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Sector',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='text-hover-bar')
import plotly.plotly as py

import plotly.graph_objs as go

trace0 = go.Bar(

    x=Subsector_list,

    y= [7377,52799,264153,772236,982,1263662,18501,2931990,1185809,152567,126237,611773,692118,4178,1517719,805884,2551735,333205,],

           

    text=['', '', ''],

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        )

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Subsector',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='text-hover-bar')