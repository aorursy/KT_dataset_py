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
import pandas as pd

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
df = pd.read_csv('../input/data.csv')

df.head()
df.columns
import cufflinks as cf

import matplotlib.pyplot as plt

cf.set_config_file(offline=False, world_readable=True, theme='ggplot')



n_df = df.sort_values(by = 'Age', ascending = False)

age = n_df['Age'].values

fig = go.Histogram(x = age)

iplot([fig])
ovr = df.sort_values(by = 'Overall', ascending=False)

top_best = ovr.loc[:4, ['Name', 'Overall']]

top_best

#arr = top_best.values

#a, b = zip(*arr)

#fig = go.Bar(x = a, y= b)

#iplot([fig])
age = df.sort_values(by = 'Age')

age.head()

rel = age.loc[:,[ 'Age','Overall']]

rel.values



fig = go.Scatter(x= rel['Age'] , y= rel['Overall'])

layout = go.Layout(title = 'Age Overall relationship')

figure = go.Figure(data=[fig], layout=layout)

iplot(figure)
pos_count = df['Position'].value_counts()

trace = go.Pie(labels = pos_count.index, values = pos_count.values, hole = 0.4)

data=[trace]

layout = go.Layout(

                    title = 'Percentage position')

fig = go.Figure(data, layout)

iplot(fig)

print ("Maxmimum Pos " + pos_count.idxmax(), pos_count.max())
all_position = df.iloc[df.groupby('Position')['Overall'].idxmax()][['Name', 'Position']]

pos = all_position['Position'].values

all_position['Name']
young_team_df = df.where(df['Age']<=22)

young_team_list = young_team_df.groupby('Position')['Overall'].idxmax()

young_team = young_team_df.iloc[young_team_list]

young_team[['Position','Name']]
counter = 0

for x in pos:

    posi = all_position['Position'].values

    name = all_position.where(all_position['Position']== x ).dropna()['Name'].values

    neymar = df[['Agility','BallControl','Strength','Acceleration','ShotPower','Vision']].where(df['Name']==name[0]).dropna()

    r = neymar.values

    r = r.astype(int)[0]

#r = np.append(r,r[0])

    theta = np.array(neymar.columns)

#theta = np.append(theta, theta[0])

#theta

    data = go.Scatterpolar(

                r = r,

                theta = theta,

                fill = 'toself')

    layout = go.Layout(

      polar = dict(

        radialaxis = dict(

          visible = True,

          range = [0, 100]

        )

      ),

        title = name[0],

      showlegend = False

    )

    fig = go.Figure(data=[data], layout=layout)

    iplot(fig)
trace1 = {

    'x' : [19],

    'y' : [25],

    'type': 'scatter',

    'name' : 'GK',

    'text': all_position['Name'].where(all_position['Position']=='GK').dropna().values,

    'marker':{'color':'pink','size':20}    

}

trace2 = {

    'x' : [25],

    'y' : [15],

    'type': 'scatter',

    'name' : 'LCB',

    'text': all_position['Name'].where(all_position['Position']=='LCB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace3 = {

    'x' : [25],

    'y' : [35],

    'type': 'scatter',

    'name' : 'RCB',

    'text': all_position['Name'].where(all_position['Position']=='RCB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace4 = {

    'x' : [30],

    'y' : [45],

    'type': 'scatter',

    'name' : 'LB',

    'text': all_position['Name'].where(all_position['Position']=='LB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace5 = {

    'x' : [30],

    'y' : [5],

    'type': 'scatter',

    'name' : 'RB',

    'text': all_position['Name'].where(all_position['Position']=='RB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace6 = {

    'x' : [40],

    'y' : [25],

    'type': 'scatter',

    'name' : 'CDM',

    'text': all_position['Name'].where(all_position['Position']=='CDM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace7 = {

    'x' : [50],

    'y' : [40],

    'type': 'scatter',

    'name' : 'LCM',

    'text': all_position['Name'].where(all_position['Position']=='LCM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace8 = {

    'x' : [50],

    'y' : [12],

    'type': 'scatter',

    'name' : 'RCM',

    'text': all_position['Name'].where(all_position['Position']=='RCM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace9 = {

    'x' : [65],

    'y' : [50],

    'type': 'scatter',

    'name' : 'LF',

    'text': all_position['Name'].where(all_position['Position']=='LF').dropna().values,

    'marker':{'color':'blue','size':20}    

}

trace10 = {

    'x' : [65],

    'y' : [2],

    'type': 'scatter',

    'name' : 'RF',

    'text': all_position['Name'].where(all_position['Position']=='RF').dropna().values,

    'marker':{'color':'blue','size':20}    

}

trace11 = {

    'x' : [60],

    'y' : [25],

    'type': 'scatter',

    'name' : 'ST',

    'text': all_position['Name'].where(all_position['Position']=='ST').dropna().values,

    'marker':{'color':'yellow','size':20}

}

Layout={

    "title":"Dream Team"

}

data=[trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11]

fig = go.Figure(data = data, layout=Layout)

iplot(fig)
trace1 = {

    'x' : [19],

    'y' : [25],

    'type': 'scatter',

    'name' : 'GK',

    'text': young_team['Name'].where(young_team['Position']=='GK').dropna().values,

    'marker':{'color':'pink','size':20}    

}

trace2 = {

    'x' : [25],

    'y' : [15],

    'type': 'scatter',

    'name' : 'LCB',

    'text': young_team['Name'].where(young_team['Position']=='LCB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace3 = {

    'x' : [25],

    'y' : [35],

    'type': 'scatter',

    'name' : 'RCB',

    'text': young_team['Name'].where(young_team['Position']=='RCB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace4 = {

    'x' : [30],

    'y' : [45],

    'type': 'scatter',

    'name' : 'LB',

    'text': young_team['Name'].where(young_team['Position']=='LB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace5 = {

    'x' : [30],

    'y' : [5],

    'type': 'scatter',

    'name' : 'RB',

    'text': young_team['Name'].where(young_team['Position']=='RB').dropna().values,

    'marker':{'color':'red','size':20}    

}

trace6 = {

    'x' : [40],

    'y' : [25],

    'type': 'scatter',

    'name' : 'CDM',

    'text': young_team['Name'].where(young_team['Position']=='CDM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace7 = {

    'x' : [50],

    'y' : [40],

    'type': 'scatter',

    'name' : 'LCM',

    'text': young_team['Name'].where(young_team['Position']=='LCM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace8 = {

    'x' : [50],

    'y' : [12],

    'type': 'scatter',

    'name' : 'RCM',

    'text': young_team['Name'].where(young_team['Position']=='RCM').dropna().values,

    'marker':{'color':'green','size':20}    

}

trace9 = {

    'x' : [65],

    'y' : [50],

    'type': 'scatter',

    'name' : 'LF',

    'text': young_team['Name'].where(young_team['Position']=='LF').dropna().values,

    'marker':{'color':'blue','size':20}    

}

trace10 = {

    'x' : [65],

    'y' : [2],

    'type': 'scatter',

    'name' : 'RF',

    'text': young_team['Name'].where(young_team['Position']=='RF').dropna().values,

    'marker':{'color':'blue','size':20}    

}

trace11 = {

    'x' : [60],

    'y' : [25],

    'type': 'scatter',

    'name' : 'ST',

    'text': young_team['Name'].where(young_team['Position']=='ST').dropna().values,

    'marker':{'color':'yellow','size':20}

}

Layout={

    "title":"Young player's team under 22"

}

data=[trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11]

fig = go.Figure(data=data, layout=Layout)

iplot(fig)