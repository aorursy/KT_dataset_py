import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import re
data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
data.head()
states = list(data['state'].unique())
len(states)
killed=[]
injured=[]
for i in states:
    s = data[(data['state']== i)]
    k = sum(s['n_killed'])
    killed.append(k)
    i = sum(s['n_injured'])
    injured.append(i)
import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Bar(
            x=states,
            y=killed,
            name = 'Killed'
    )
trace2 = go.Bar(
            x=states,
            y=injured,
            name = 'Injured'
    )

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

iplot(go.Figure(data=data, layout=layout))

#TOTAL CASUALITIES IS NUMBER OF DEATHS + NUMBER OF INJURIES
total =[x + y for x, y in zip(killed, injured)]
data = [go.Bar(x=states,y=total)]
iplot(go.Figure(data=data))
data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
i = data.loc[data['state'] == 'Illinois',:]
city = list(i['city_or_county'].unique())
illinois_killed = {}
illinois_injured = {}
killed=[]
injured=[]
for j in city:
    s = i[(i['city_or_county']== j)]
    k = sum(s['n_killed'])
    illinois_killed[j] = k
    kk = sum(s['n_injured'])
    illinois_injured[j] = kk

illinois_killed.values()
i_k = {x:y for x,y in illinois_killed.items() if y>10}
data = [go.Bar(x=list(i_k.keys()),
            y=list(i_k.values()) ,
            marker=dict(color='#cc33ff'))]

iplot(go.Figure(data=data))

data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
city = data['city_or_county'].value_counts()
city = city[:25]
d = [go.Bar(x=list(city.index),y=list(city.values),marker=dict(color='#ffff4d'))]
iplot(go.Figure(data=d))
guns = data['gun_type']
guns = guns.dropna()
guns = [x for x in guns if x != '0::Unknown' and x!='0:Unknown']
allguns=[]
for i in guns:
    result = re.sub("\d+::", "", i)
    result = re.sub("\d+:", "", result)
    result = result.split("|")
    for j in result:
        allguns.append(j)
allguns = [x for x in allguns if x != 'Unknown']
allguns = [x for x in allguns if x]
from collections import Counter
allguns = Counter(allguns)
labels, values = zip(*allguns.items())
d = [go.Bar(x=list(labels),y=list(values),marker=dict(color='#ff0055'))]
iplot(go.Figure(data=d))
data= pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
d = data.groupby('date').sum()

d.head()
data = [go.Scatter(
          x=d.index,
          y=d['n_killed'])]

killings = go.Scatter(
                x=d.index,
                y=d['n_killed'],
                name = "Killed",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

injured = go.Scatter(
                x=d.index,
                y=d['n_injured'],
                name = "injured",
                line = dict(color = '#7F7F7F'),
                opacity = 0.8)

data = [killings,injured]

iplot(go.Figure(data=data))

