# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
# Importing data
data = pd.read_csv('../input/pokemon.csv')
copied_data = data.copy()
data.info()
data.isnull().sum()
data.sample(10)
# Droping "#" column
data.drop(['#'], axis=1, inplace = True)
# Starting index from 1 & assigning an index name
data.index = range(1,801,1)
data.index.name = "New Index"       
data.head(10) # let's check it now.

# An alternative and easy way to do the same thing:
#data.set_index('#', inplace = True)
# Correlation map through heatmap
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr('spearman'),linewidths=1, linecolor='black', cmap='Reds', annot = True, fmt='.2f',ax=ax)
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.show()              # you don't need to write this, but if you don't write, an information script will be show up and visually ruins your kernel.
trace1 =go.Scatter(
                    x = data.Attack,
                    y = data.Defense,
                    mode = "markers",
                    name = "Attack",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= data['Type 1'])

trace2 =go.Scatter(
                    x = data.Attack,
                    y = data.Speed,
                    mode = "markers",
                    name = "Defense",
                    marker = dict(color = 'rgba(15, 200, 30, 0.4)'),
                    text= data['Type 1'])

data2 = [trace1, trace2]
layout = dict(title = 'Defense and Speed values with respect to Attack',
              xaxis= dict(title= 'Attack',ticklen= 5,zeroline= False), # ticklen : eksenlerdeki değerlerin ticklerinin uzunluğu
              yaxis= dict(title= 'Defense & Speed',ticklen= 5,zeroline= False)
             )
fig = dict(data = data2, layout = layout)
iplot(fig)
plt.figure(figsize=(10,10))
sns.barplot(x= data['Type 1'].value_counts().index, y= data['Type 1'].value_counts().values)
plt.xticks(rotation=45)
plt.show()
bar = go.Bar(
                x= data['Type 1'].value_counts().index,
                y= data['Type 1'].value_counts().values,
                marker = dict(color = 'rgba(21, 180, 255, 0.7)',
                             line=dict(color='rgb(104,32,0)',width=1.5)),
                text = data['Type 1'].value_counts().index)
databar = [bar]
layout = dict(title = 'Value Counts of Type 1 Pokemons',
             xaxis =dict(title='Species' ),
              yaxis =dict(title='Counts' )
             )
fig = go.Figure(data = databar, layout = layout)
iplot(fig)
data_new = data.copy()
data_new.dropna(subset=['Type 2'], inplace = True)
data_new.index = range(1,415,1)
data_new.index.name = 'Dropped'
data_new.head()
fig = {
  "data": [
    {
      "values": data_new['Type 2'].value_counts().values,
      "labels": data_new['Type 2'].value_counts().index,
      "type": "pie"
    }],
  "layout": {
        "title":"Percentages of Legendary Pokemons w.r.t. Types"
    }
}
iplot(fig)
data.sort_values('Attack',inplace=True,ascending=False)
datarank = data.copy()
datarank.index = range(0,800,1)

nums = copied_data['#'].iloc[:50]

sorted_data = pd.concat([datarank,nums], axis=1).iloc[:50]
sorted_data.head()
data_bubble = [ dict(x=sorted_data['#'],
         y=sorted_data['Attack'],
         mode= 'markers',
         marker = dict(size = datarank.Defense/3, color = datarank.Speed, showscale = True),
                 
         text = sorted_data.Name)]
iplot(data_bubble)
data['Type 1'].nunique()
types = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 255, 18)]

for i in range(18):
    violins = {
            "type": 'violin',
            "y": data.Attack[data['Type 1'] == data['Type 1'].value_counts(ascending=False).index[i]],
            "name": data['Type 1'].value_counts(ascending=False).index[i],
            "marker":{
                "color":c[i]},
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    types.append(violins)
iplot(types)
# Splitting Data
data_water = data[data['Type 1']=='Water']
data_grass = data[data['Type 1']=='Grass']
data_fire = data[data['Type 1']=='Fire']
data_bug = data[data['Type 1']=='Bug']
data_psychic = data[data['Type 1']=='Psychic']

box1 = go.Box(
                y= data_water.Attack,
                name= 'Water Pokemons',
                marker = dict(color = 'rgb(12, 128, 128)'),
                boxmean='sd',
                boxpoints='all')
box2 = go.Box(
                y= data_grass.Attack,
                name= 'Grass Pokemons',
                marker = dict(color = 'rgb(100, 12, 38)'),
                boxmean='sd',
                boxpoints='all')
box3 = go.Box(
                y= data_fire.Attack,
                name= 'Fire Pokemons',
                marker = dict(color = 'rgb(12, 128, 128)'),
                boxmean='sd',
                boxpoints='all')
box4 = go.Box(
                y= data_bug.Attack,
                name= 'Bug Pokemons',
                marker = dict(color = 'rgb(50, 40, 100)'),
                boxmean='sd',
                boxpoints='all')
box5 = go.Box(
                y= data_psychic.Attack,
                name= 'Psychic Pokemons',
                marker = dict(color = 'rgb(45, 179, 66)'),
                boxmean='sd',
                boxpoints='all')

data_boxes = [box1,box2,box3,box4,box5]
iplot(data_boxes)