# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from wordcloud import WordCloud

import plotly.graph_objs as go

import seaborn as sns
fifa19=pd.read_csv("../input/data.csv")

fifa19.head(8)
#float64(38), int64(6), object(45)

fifa19.info()

print(50*('-'))

fifa19.isnull().any()
columnsnotnull = [

    'Name',

    'Age',

    'Nationality',

    'Overall',

    'Potential',

    'Special',

    'Acceleration',

    'Aggression',

    'Agility',

    'Balance',

    'BallControl',

    'Body Type',

    'Composure',

    'Crossing',

    'Curve',

    'Club',

    'Dribbling',

    'FKAccuracy',

    'Finishing',

    'GKDiving',

    'GKHandling',

    'GKKicking',

    'GKPositioning',

    'GKReflexes',

    'HeadingAccuracy',

    'Interceptions',

    'International Reputation',

    'Jersey Number',

    'Jumping',

    'Joined',

    'LongPassing',

    'LongShots',

    'Marking',

    'Penalties',

    'Position',

    'Positioning',

    'Preferred Foot',

    'Reactions',

    'ShortPassing',

    'ShotPower',

    'Skill Moves',

    'SlidingTackle',

    'SprintSpeed',

    'Stamina',

    'StandingTackle',

    'Strength',

    'Value',

    'Vision',

    'Volleys',

    'Wage',

    'Weak Foot',

    'Work Rate'

]
fifa19df=pd.DataFrame(fifa19, columns = columnsnotnull)

fifa19df.head()
fifa19a=fifa19df.loc[0:100,:]

fifa19a.head()


trace1=go.Scatter(x=fifa19a.index.values,

                  y=fifa19a.Potential,

                  mode="lines+markers",

                name="potential",

                 marker=dict(color='rgba(16,112,2,0.8)'),

                  text=fifa19a.Name)

trace2=go.Scatter(x=fifa19a.index.values,

                 y=fifa19a.Overall,

                 mode="lines+markers",

                 name="overall",

                 marker=dict(color='rgba(80,26,80,0.8)'),

                 text=fifa19a.Name)

data=[trace1,trace2]

layout=dict(title='Potential and Overall Power',

           xaxis=dict(title="İndex",ticklen=10,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
position19=list(fifa19a.Position.unique())

position19
meanposition19=[]

for each in position19:

    x=fifa19a[fifa19a.Position==each]

    mean=sum(x.Potential)/len(x)

    meanposition19.append(mean)

meanposition19



dictionarymp={"Position":position19,"Mean":meanposition19}

datamp=pd.DataFrame(dictionarymp)



trace1=go.Scatter(x=datamp.Position,y=datamp.Mean,mode="lines+markers",marker=dict(color="rgba(0,255,0,0.8)"),text=datamp.Position)

data=[trace1]

layout=dict(title='Mean Potential Each Position',

           xaxis=dict(title="Position",ticklen=10,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
trace1=go.Scatter(x=fifa19a.index.values,y=fifa19a.Curve,mode="lines+markers",marker=dict(color="rgba(0,255,0,0.8)"),text=fifa19a.Name)

data=[trace1]

layout=dict(title='Curve for Each Football Player',

           xaxis=dict(title="Index of Player",ticklen=10,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
curvemean19=[]

for each in position19:

    x=fifa19a[fifa19a.Position==each]

    mean=sum(x.Curve)/len(x)

    curvemean19.append(mean)

dictionarycm={"Position":position19,"Mean":curvemean19}

datacm=pd.DataFrame(dictionarycm)



trace1=go.Scatter(x=datacm.Position,y=datacm.Mean,mode="lines+markers",marker=dict(color="rgba(0,255,0,0.8)"),text=datacm.Position)

data=[trace1]

layout=dict(title='Curve for Each Position',

           xaxis=dict(title="Positions",ticklen=10,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
clublist=list(fifa19a.Club.unique())

clubcount=[]

for each in clublist:

    x=fifa19a[fifa19a.Club==each]

    count=len(x.Club)

    clubcount.append(count)

dictionarycc={"Club Name":clublist,"Player Count":clubcount}

datacc=pd.DataFrame(dictionarycc)



trace1=go.Scatter(x=datacc.index.values,y=datacc["Player Count"],mode="lines+markers",marker=dict(color="rgba(0,255,0,0.8)"),text=datacc["Club Name"])

data=[trace1]

layout=dict(title='Club Count in Top100',

           xaxis=dict(title="Positions",ticklen=10,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
trace1=go.Bar(x=datacc["Club Name"],y=datacc["Player Count"],marker=dict(color="rgba(0,255,255,0.8)", line=dict(color='rgb(0,0,0)',width=1.5)),text=datacc["Club Name"])

data=[trace1]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# SORTED DATA

datacc.head()

new_index = (datacc["Player Count"].sort_values(ascending=False)).index.values

sorted_data =datacc.reindex(new_index)

trace1=go.Bar(x=sorted_data["Club Name"],y=sorted_data["Player Count"],marker=dict(color="rgba(0,255,255,0.8)", line=dict(color='rgb(0,0,0)',width=1.5)),text=sorted_data["Club Name"])

data=[trace1]

layout = go.Layout(barmode = "relative")

fig = go.Figure(data = data, layout = layout)

iplot(fig)

#trace1 = {

 # 'x': x,

  #'y': datacc.,

  #'name': '',

  #'type': 'bar'

#};
#trace1=go.Pie(labels=sorted_data["Club Name"],values=sorted_data["Player Count"])

fig = {

  "data": [

    {

      "values": sorted_data["Player Count"],

      "labels": sorted_data["Club Name"],

      "domain": {"x": [0, .5]},

      "name": "Per Count",

      "hoverinfo":"label+percent+name",

      "textinfo":"value",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie Chart",

        "annotations": [

            { "font": { "size": 15},

              "showarrow": False,

                 "text":"",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
#trace1=go.Pie(labels=sorted_data["Club Name"],values=sorted_data["Player Count"])

fig = {

  "data": [

    {

      "values": sorted_data["Player Count"],

      "labels": sorted_data["Club Name"],

      "domain": {"x": [0, .5]},

      "name": "Per Count",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie Chart",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

                 "text":"",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
datacc.head()
#Bubble Chart

#trace1=go.Scatter(x=datacc.index.values,y=datacc["Player Count"],mode="lines+markers",marker=dict(color="rgba(0,255,0,0.8)"),

                  #text=datacc["Club Name"])



data=[{"x":datacc["Club Name"],"y":datacc["Player Count"],

       "mode":"markers",

       "marker":{"color":datacc["Player Count"],"size":5*datacc["Player Count"],"showscale":True},"text":datacc["Club Name"]}]

iplot(data)
fifa19a.head()
preferfoot=list(fifa19a["Preferred Foot"].value_counts())

plt.figure(figsize=(10,10))

sns.countplot(fifa19a["Preferred Foot"])

plt.show()
#fifa19not=fifa19a[fifa19a.Position!='GK']

#preferleft=fifa19not[fifa19not["Preferred Foot"]=="Left"]

#preferright=fifa19not[fifa19not["Preferred Foot"]=="Right"]

#x=preferleft.iloc[:,10]

#a=pd.DataFrame(x)

#new_index = (datacc["Player Count"].sort_values(ascending=False)).index.values

#sorted_data =datacc.reindex(new_index)

#index_value=(a["BallControl"].sort_values(ascending=False)).index.values

#sortvalue=a.reindex(index_value)

#sns.barplot(x=sortvalue.index.values,y="BallControl",data=sortvalue)
print(np.mean(fifa19a["BallControl"][fifa19a["Preferred Foot"]=="Left"]))

print(np.mean(fifa19a["BallControl"][fifa19a["Preferred Foot"]=="Right"]))