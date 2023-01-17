# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library

import matplotlib.pyplot as plt # visualization library

import plotly as py # visualization library

from wordcloud import WordCloud

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
country_wise=pd.read_csv("../input/corona-virus-report/country_wise_latest.csv")#country wise 

day_wise=pd.read_csv("../input/corona-virus-report/day_wise.csv")#day wise 

country_wise.info()
country_wise.head()
mostconfirmed=country_wise[country_wise.Confirmed>200000]

labels = mostconfirmed['Country/Region']

confirmed=  mostconfirmed['Confirmed']

fig = {

  "data": [

    {

      "values": confirmed,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of confirmed cases",

      "hoverinfo":"label+ value",

      "hole": .1,

      "type": "pie"

    },],

  "layout": {

        "title":"Number Of confirmed cases",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Rate Of confirmed cases",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
sortedcountrywise=country_wise.sort_values('Confirmed',ascending=False)

top20sortedcountrywise=sortedcountrywise.iloc[:20,:]



normalizedconfirmed=top20sortedcountrywise['Confirmed']/100000

normalizeddeath=top20sortedcountrywise['Deaths']/10000


trace1= go.Bar(

                    x=top20sortedcountrywise['Country/Region'],

                    y=normalizedconfirmed,

                   # mode="markers",

                    name="Confirmed cases(*100k)",

                    marker=dict(

                        color='rgba(10,10,200,0.6)'),

                    text=top20sortedcountrywise['Country/Region'],

)

trace2= go.Bar(

                    x=top20sortedcountrywise['Country/Region'],

                    y=normalizeddeath,

                    #mode="markers",

                    name="Number of deaths(*10k)",

                    marker=dict(

                        color='rgba(200,10,10,0.6)'),

                    text=top20sortedcountrywise['Country/Region'],

)

data=[trace1,trace2]

layout= dict(

            title="Confirmed cases and deaths in countries",

            xaxis= dict(title= 'Countries',ticklen= 5,zeroline= False)

)

fig=dict(data=data,layout=layout)

iplot(fig)
day_wise.head()
trace1= go.Scatter(

                    x=day_wise.Date,

                    y=day_wise.Confirmed,

                    mode="lines",

                    name="Total Confirmed cases",

                    marker=dict(

                        color='rgba(10,10,200,0.6)'),

                    text= day_wise.Confirmed ,

)

trace2= go.Scatter(

                    x=day_wise.Date,

                    y=day_wise.Active,

                    mode="lines",

                    name="Active cases",

                    marker=dict(

                        color='rgba(100,200,100,0.6)'),

                    text= day_wise.Confirmed ,

)

data=[trace1,trace2]

layout=dict(title="        Total confirmed cases in the world",

            xaxis= dict(title= 'Dates',ticklen= 10,zeroline= False))

fig=dict(data=data,layout=layout)

iplot(fig)
trace1= go.Scatter(

                    x=day_wise.Date,

                    y=day_wise['Deaths / 100 Cases'],

                    mode="markers",

                    name="Deaths / 100 Cases",

                    marker=dict(

                        color='rgba(10,10,200,0.6)'),

                    text= day_wise.Deaths,

)

data=[trace1]

layout=dict(title="Deaths / 100 Cases in the world ",

            xaxis= dict(title= 'Dates',ticklen= 10,zeroline= False))

fig=dict(data=data,layout=layout)

iplot(fig)
#country_wise[country_wise['Country/Region']=='United Kingdom'].Recovered=int(country_wise[country_wise['Country/Region']=='United Kingdom'].Confirmed*88/100)

#uk=country_wise[country_wise['Country/Region']=='United Kingdom']

#uk.head()
most30countries=country_wise.sort_values('Confirmed',ascending=False).iloc[:30,:]

most30countries.info()
color=most30countries.Confirmed*255/most30countries.Confirmed.max()
color=most30countries.Confirmed*100/most30countries.Confirmed.max()

data=[

    {

        'x': most30countries['Recovered'],

        'y': most30countries['Deaths'],

        'mode': 'markers',

        'marker':{

            'color':color,

            'size' :color

            

           #'showscale': True,

        },

        "text" : most30countries['Country/Region']

    

    }

]

layout=dict(title="Recovered and Death people according to countries",

            xaxis= dict(title= 'Recovery',ticklen= 10,zeroline= False),

            yaxis= dict(title= 'Death',ticklen= 10,zeroline= False))

fig=dict(data=data,layout=layout)

iplot(fig)
trace0= go.Box(

    y=most30countries['Deaths / 100 Cases'],

    name = 'Deaths / 100 Cases',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    ),

    text = most30countries['Country/Region'],

)

trace1= go.Box(

    y=most30countries['Recovered / 100 Cases'],

    name = 'Recovered / 100 Cases',

    marker = dict(

        color = 'rgb(12, 120, 150)',

    ),

    text = most30countries['Country/Region'],

)

data = [trace0, trace1]

iplot(data)
import plotly.figure_factory as ff

datas= most30countries.loc[:,["Deaths","Recovered","Active","Confirmed"]]

#datas=datas.iloc[1:,:]

datas["index"] = np.arange(1,len(datas)+1)

fig = ff.create_scatterplotmatrix(datas, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
country_list=list(most30countries['Country/Region'])

# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=most30countries.Confirmed,y=country_list,color='green',alpha = 0.5,label='Confirmed cases' )

sns.barplot(x=most30countries.Deaths ,y=country_list,color='red',alpha = 0.5,label='number of deaths' )

sns.barplot(x=most30countries.Recovered ,y=country_list,color='blue',alpha = 0.5,label='recovered people' )

sns.barplot(x=most30countries.Active ,y=country_list,color='orange',alpha = 0.5,label='active cases' )



ax.legend(loc='lower right',frameon = True) 

ax.set(xlabel='Number of cases', ylabel='Countries',title = "The number of Covid-19 cases  ")

plt.show()
correlationmost30= most30countries.loc[:,("Confirmed","Deaths","Recovered","Active")]

f,az=plt.subplots(figsize=(6,6))

sns.heatmap(correlationmost30.corr(), annot=True, linewidths=0.6, linecolor="blue", fmt='.2f', ax=az)

plt.show()