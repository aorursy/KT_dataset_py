import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF





import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

d.index = pd.DatetimeIndex(d.timeStamp)

d=d[(d.timeStamp >= "2016-01-01 00:00:00")]



# Get Date from DateTime

d['date']=d['timeStamp'].dt.date
d.head(6)


sns.color_palette("cubehelix", 8)

sns.set_style("whitegrid", {'axes.grid' : False})



gtr=d[(d['title']=='Traffic: HAZARDOUS ROAD CONDITIONS -')].groupby(['date']).size().reset_index()

gtr.columns = ['date','count']



gta=d[(d['title']=='Traffic: VEHICLE ACCIDENT -')].groupby(['date']).size().reset_index()

gta.columns = ['date','count']



gtf=d[(d['title']=='Traffic: VEHICLE FIRE -')].groupby(['date']).size().reset_index()

gtf.columns = ['date','count']









sns.color_palette("cubehelix", 8)

sns.distplot(gtr['count'],bins=100,hist=False,   label="Traffic: HAZARDOUS ROAD CONDITIONS -");

sns.distplot(gta['count'],bins=100,hist=False,   label="Traffic: VEHICLE ACCIDENT -");

sns.distplot(gtf['count'],bins=100,hist=False,   label="Traffic: VEHICLE FIRE -");









plt.legend();
# You need this module 

# from plotly.tools import FigureFactory as FF



d['hour'] = d['timeStamp'].apply(lambda x: x.hour)



gtr=d[(d['title']=='Traffic: HAZARDOUS ROAD CONDITIONS -')].groupby(['hour']).size().reset_index()

gtr.columns = ['hour','count']

#gtr.head()



x = gtr['hour']

y = gtr['count']



colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]



fig = FF.create_2D_density(

    x, y, colorscale=colorscale,

    title='Traffic: HAZARDOUS ROAD CONDITIONS',

    hist_color='rgb(255, 237, 222)', point_size=3

)

iplot(fig)
# Example of normalized Histogram



v0 = d[(d['title']=='Traffic: VEHICLE ACCIDENT -')].hour.values



data = [

    go.Histogram(

        x=v0,

        histnorm='probability'

    )

]





layout = dict(

            title='Traffic: VEHICLE ACCIDENT (hr)',

            autosize= True,

            bargap= 0.015,

            height= 400,

            width= 500,       

            hovermode= 'x',

            xaxis=dict(

            autorange= True,

            zeroline= False),

            yaxis= dict(

            autorange= True,

            showticklabels= True,

           ))



fig1 = dict(data=data, layout=layout)





iplot(fig1)
v0 = d[(d['title']=='Traffic: VEHICLE ACCIDENT -')].hour.values

v1 = d[(d['title']=='EMS: VEHICLE ACCIDENT')].hour.values

v2 = d[(d['title']=='Traffic: VEHICLE FIRE -')].hour.values





data = [

    go.Histogram(

        x=v0,

        histnorm='probability',

        name='Traffic: VEHICLE ACCIDENT'



    ),

    

    go.Histogram(

        x=v1,

        histnorm='probability',

        name='EMS: VEHICLE ACCIDENT'

        

    ),

    

    go.Histogram(

        x=v2,

        histnorm='probability',

        name='Traffic: VEHICLE FIRE -'

    )

]





layout = dict(

            title='VEHICLE ACCIDENTS',

            autosize= True,

            bargap= 0.015,

            height= 500,

            width= 700,       

            hovermode= 'x',

            xaxis=dict(

            autorange= True,

            zeroline= False),

            yaxis= dict(

            autorange= True,

            showticklabels= True,

           ))



fig1 = dict(data=data, layout=layout)





iplot(fig1)
# Create a pivot table

#  d['c'] = 1 is a dummy argument



d['c'] = 1

g=pd.pivot_table(d, values='c', index=['twp'],

            columns=['title'], aggfunc=np.sum,fill_value=0).reset_index()



"""

Traffic: VEHICLE ACCIDENT -

Traffic: HAZARDOUS ROAD CONDITIONS -

"""

twp = 'ABINGTON'

a0=g[g['twp']==twp]['Traffic: VEHICLE ACCIDENT -'].values[0]

a1=g[g['twp']==twp]['Traffic: HAZARDOUS ROAD CONDITIONS -'].values[0]

a2=g[g['twp']==twp]['Fire: FIRE ALARM'].values[0]

a3=g[g['twp']==twp]['EMS: RESPIRATORY EMERGENCY'].values[0]



twp = 'CHELTENHAM'

c0=g[g['twp']==twp]['Traffic: VEHICLE ACCIDENT -'].values[0]

c1=g[g['twp']==twp]['Traffic: HAZARDOUS ROAD CONDITIONS -'].values[0]

c2=g[g['twp']==twp]['Fire: FIRE ALARM'].values[0]

c3=g[g['twp']==twp]['EMS: RESPIRATORY EMERGENCY'].values[0]





fig = {

  "data": [

    {

      "values": [a0,a1,a2,a3],

      "labels": [

        "VEHICLE ACCIDENT",

        "HAZARDOUS ROAD CONDITIONS",

        "FIRE ALARM",

        "RESPIRATORY EMERGENCY",

        

      ],

      "domain": {"x": [0, .48]},

      "name": "Abington",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": [c0,c1,c2,c3],

      "labels": [

        "VEHICLE ACCIDENT",

        "HAZARDOUS ROAD CONDITIONS",

        "FIRE ALARM",

        "RESPIRATORY EMERGENCY",

        

      ],

      "text":"CHLT",

      "textposition":"inside",

      "domain": {"x": [.52, 1]},

      "name": "Cheltenham",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Abington vs Cheltenham",

        "annotations": [

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Abington",

                "x": 0.194,

                "y": 0.5

            },

            {

                "font": {

                    "size": 10

                },

                "showarrow": False,

                "text": "Cheltenham",

                "x": 0.83,

                "y": 0.5

            }

        ]

    }

}

iplot(fig)





trace0 = go.Scatter(

    x=[1, 2, 3, 4],

    y=[10, 11, 12, 13],

    text=['A</br>size: 40</br>default', 'B</br>size: 60</br>default', 'C</br>size: 80</br>default', 'D</br>size: 100</br>default'],

    mode='markers',

    name='Trace 0',

    marker=dict(

        size=[400, 600, 800, 1000],

        sizemode='area',

    )

)

trace1 = go.Scatter(

    x=[1, 2, 3, 4],

    y=[14, 15, 16, 17],

    text=['A</br>size: 40</br>sixeref: 0.6', 'B</br>size: 60</br>sixeref: 0.6', 'C</br>size: 80</br>sixeref: 0.6', 'D</br>size: 100</br>sixeref: 0.6'],

    mode='markers',

    name='Trace 1',

    marker=dict(

        size=[400, 600, 800, 1000],

        sizeref=0.6,

        sizemode='area',

    )

)

trace2 = go.Scatter(

    x=[1, 2, 3, 4],

    y=[20, 21, 22, 23],

    text=['A</br>size: 40</br>sixeref: 0.2', 'B</br>size: 60</br>sixeref: 0.2', 'C</br>size: 80</br>sixeref: 0.2', 'D</br>size: 100</br>sixeref: 0.2'],

    mode='markers',

    name='Trace 2',

    marker=dict(

        size=[400, 600, 800, 1000],

        sizeref=0.2,

        sizemode='area',

    )

)





trace2 = go.Scatter(

    x=[1, 2, 3, 4],

    y=[20, 21, 22, 23],

    text=['A</br>size: 40</br>sixeref: 0.2', 'B</br>size: 60</br>sixeref: 0.2', 'C</br>size: 80</br>sixeref: 0.2', 'D</br>size: 100</br>sixeref: 0.2'],

    mode='markers',

    name='Trace 2',

    marker=dict(

        size=[400, 600, 800, 1000],

        sizeref=0.2,

        sizemode='area',

    )

)



trace4 = go.Scatter(

    x = [1,2,3,4],

    y = [25,16,3,19],

    text=['Value 1<br>With some extra<br>text',

          'Value 2','Value 3','Value 4<br><br>This is a lot more text here<br><br>'],

    name='Trace 4',

)



trace5 = go.Bar(                # all "bar" chart attributes: /python/reference/#bar

        x=[1, 2, 3],            # more about "x": /python/reference/#bar-x

        y=[3, 1, 6],            # /python/reference/#bar-y

        name="Bar Options"      # /python/reference/#bar-name

    )



trace6 = go.Bar(                # all "bar" chart attributes: /python/reference/#bar

        x=[1, 2, 3],            # more about "x": /python/reference/#bar-x

        y=[3, 1, 6],            # /python/reference/#bar-y

        name="Bar Options"      # /python/reference/#bar-name

    )









data = [trace0, trace1, trace2, trace4,trace5,trace6]



iplot(data)
import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF









import plotly.graph_objs as graph_objs





mapbox_access_token = 'pk.eyJ1IjoiY2hlbHNlYXBsb3RseSIsImEiOiJjaXFqeXVzdDkwMHFrZnRtOGtlMGtwcGs4In0.SLidkdBMEap9POJGIe1eGw'



data = graph_objs.Data([

    graph_objs.Scattermapbox(

        lat=['45.5017'],

        lon=['-73.5673'],

        mode='markers',

    )

])

layout = graph_objs.Layout(

    height=600,

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        layers=[

            dict(

                sourcetype = 'geojson',

                source = 'https://raw.githubusercontent.com/plotly/datasets/master/florida-red-data.json',

                type = 'fill',

                color = 'rgba(163,22,19,0.8)'

            ),

            dict(

                sourcetype = 'geojson',

                source = 'https://raw.githubusercontent.com/plotly/datasets/master/florida-blue-data.json',

                type = 'fill',

                color = 'rgba(40,0,113,0.8)'

            )

        ],

        accesstoken= mapbox_access_token,

        bearing=0,

        center=dict(

            lat=27.8,

            lon=-83

        ),

        pitch=0,

        zoom=5.2,

        style='light'

    ),

)



fig = dict(data=data, layout=layout)

iplot(fig, filename='county-level-choropleths-python')