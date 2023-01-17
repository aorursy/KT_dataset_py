import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import seaborn as sns

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

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
df = pd.read_csv("../input/2015_Greenhouse_Gas_Report-_Data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_').str.replace(',', '_')

df.drop(["source", "address", "location_latitude__longitude"], axis = 1, inplace = True)
county_list = list(df['county'].unique())

county_list.sort() 

  

print(county_list) 
sector_list = list(df['sector'].unique())

sector_list.sort() 

  

print(sector_list) 
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=df.sector        ,

    y=df.county       ,

    z=df.total_emissions_mt_co2e           ,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
import plotly.plotly as py

from plotly.graph_objs import *





labels = df.clean_air_rule

pie1_list=df.index



 

labels1 = df.energy_intensive__trade_exposed

pie1_list1=df.index



fig = {

    'data': [

        {

            'labels': labels,

            'values': pie1_list,

            'type': 'pie',

            'name': 'Clean Air Rule',

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

            'labels': labels1,

            'values': pie1_list1,

            'marker': {'colors': ['rgb(177, 127, 38)',

                                  'rgb(205, 152, 36)',

                                  'rgb(99, 79, 37)',

                                  'rgb(129, 180, 179)',

                                  'rgb(124, 103, 37)']},

            'type': 'pie',

            'name': 'Energy Intensive Trade Exposed',

            'domain': {'x': [.52, 1],

                       'y': [0, .49]},

            'hoverinfo':'label+percent+name',

            'textinfo':'none'



        },

        {

            'labels': sector_list,

            'values': [424603,698676,350428,522082,1879715,806094,586304,11733778,7927026,6195994,1096820,1167584],

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

            'labels': county_list,

            'values': [165252,256023,331207,266438,1636918,4244560,31208,100011,342433,2621964,13388,572693,1089175,62453,12838,941435,6400205,41324,46229,44680,2197314,3851095,12443,243937,429460,999321,1070490,5036790,142175,185645],

         

            'marker': {'colors': ['rgb(146, 123, 21)',

                                  'rgb(177, 180, 34)',

                                  'rgb(206, 206, 40)',

                                  'rgb(175, 51, 21)',

                                  'rgb(35, 36, 21)']},

            'type': 'pie',

            'name':'County',

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
trace1 = go.Scatter(

    x=county_list           ,

    y=[0,0,0,257014,502054,2727778,0,0,3636,1390853,0,503134,1871,0,0,113660,138285,40641,45632,41661,1047425,309085,0,86443,139454,981477,717427,0,0,0],                             

    name = "Biogenic CO2 (MT CO2)"

)

trace2 = go.Scatter(

    x=county_list          ,

    y=[83616,54507,254025,1633,929178,1307975,0,85798,258138,1218099,13373,66373,1023097,62383,12822,571414,6212495,140,0,2452,1020540,3518604,10325,132250,284594,4732,255795,4233908,139742,30661],      

    xaxis='x2',

    yaxis='y2',

    name = "Fossil CO2 (MT CO2)"

)

trace3 = go.Scatter(

    x=county_list          ,

    y=[165252,256023,331207,266438,1636918,4244560,31208,100011,342433,2621964,13388,572693,1089175,62453,12838,941435,6400205,41324,46229,44680,2197314,3851095,12443,243937,429460,999321,1070490,5036790,142175,185645],

          

    xaxis='x3',

    yaxis='y3',

    name = "Total Emissions (MT CO2e)"

)

trace4 = go.Scatter(

    x=county_list          ,

    y=[81590,38314,14,4848,12322,196122,31208,1134,56736,6455,7,578,63838,31,7,255375,18774,78,86,83,123304,12196,2113,24179,2066,1886,67652,28735,2354,144684],                            

    xaxis='x4',

    yaxis='y4',

    name = "Methane (MT CO2e)"

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.35]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.35],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 0.90]

    ),

    yaxis4=dict(

        domain=[0.55, 0.90],

        anchor='x4'

    ),

    title = ''

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_expose = df.sector [df.energy_intensive__trade_exposed    == "YES"  ]        

df_clean_air_rule  = df.sector [df.clean_air_rule    == "YES"   ]       

trace1 = go.Histogram(

    x=df_expose,

    opacity=0.75,

    name = "Energy Intensive Trade Exposed",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=df_clean_air_rule,

    opacity=0.75,

    name = "Clean Energy Rule",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))







data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title='Intensive Trade Exposed, Follow Clean Energy Rule & Sectors  ',

                   xaxis=dict(title=''),

                   yaxis=dict( title='Factories'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)