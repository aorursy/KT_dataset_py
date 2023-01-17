# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data that we will use.
pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon
# information about pokemons
pokemon.info()
# shape gives number of rows and columns in a tuple
pokemon.shape
pokemon.columns
pokemon.rename(columns={"Sp. Atk":"sp_atk","Sp. Def":"sp_def"}, inplace=True)
pokemon.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in pokemon.columns]
pokemon.columns
pokemon.columns = [each.lower() for each in pokemon.columns]
pokemon.columns
#pokemon.drop('#', axis = 1, inplace = True)
pokemon.sort_values('total',ascending=False,inplace=True)
pokemon.index = [i for i in range(0,800)]
pokemon.head()
pokemon['pk_number'] = [i+1 for i in pokemon.index]
#pokemon['pk_number'] = np.arange(1,len(pokemon)+1)
# Display positive and negative correlation between columns
pokemon.corr()
#sorts all correlations with ascending sort.
pokemon.corr().unstack().sort_values().drop_duplicates()
#correlation map
plt.subplots(figsize=(10,10))
sns.heatmap(pokemon.corr(), annot=True, linewidth=".5", cmap="YlGnBu", fmt=".2f")
plt.show()
#figsize - image size
#data.corr() - Display positive and negative correlation between columns
#annot=True -shows correlation rates
#linewidths - determines the thickness of the lines in between
#cmap - determines the color tones we will use
#fmt - determines precision(Number of digits after 0)
#if the correlation between the two columns is close to 1 or 1, the correlation between the two columns has a positive ratio.
#if the correlation between the two columns is close to -1 or -1, the correlation between the two columns has a negative ratio.
#If it is close to 0 or 0 there is no relationship between them.
pokemon.describe()
pokemon.head()
pokemon.tail()
pokemon.sample(5)
pokemon.dtypes
# prepare data frame
pk = pokemon.iloc[:100,:]

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = pk.pk_number,
                    y = pk.sp_atk,
                    mode = "lines",
                    name = "sp_atk",
                    marker = dict(color = 'rgba(160, 112, 2, 0.8)'),
                    text= pk.name)
# Creating trace2
trace2 = go.Scatter(
                    x = pk.pk_number,
                    y = pk.sp_def,
                    mode = "lines+markers",
                    name = "sp_def",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= pk.name)
data = [trace1, trace2]
layout = dict(title = 'Speacial Attack and Speacial Defense of Top 100 Pokemons',
              xaxis= dict(title= 'Pokemon Number',ticklen= 10,zeroline= True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# prepare data frames
pk1 = pokemon[pokemon.generation == 1].iloc[:100,:]
pk1['pk_number'] = [i for i in range(1,101)]

pk2 = pokemon[pokemon.generation == 2].iloc[:100,:]
pk2['pk_number'] = [i for i in range(1,101)]

pk3 = pokemon[pokemon.generation == 3].iloc[:100,:]
pk3['pk_number'] = [i for i in range(1,101)]

pk4 = pokemon[pokemon.generation == 4].iloc[:100,:]
pk4['pk_number'] = [i for i in range(1,101)]

pk5 = pokemon[pokemon.generation == 5].iloc[:100,:]
pk5['pk_number'] = [i for i in range(1,101)]

# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter(
                    x = pk1.pk_number,
                    y = pk1.total,
                    mode = "markers",
                    name = "generation 1",
                    marker = dict(color = 'rgba(83, 37, 85, 1)'),
                    text= pk1.name)
# creating trace2
trace2 =go.Scatter(
                    x = pk2.pk_number,
                    y = pk2.total,
                    mode = "markers",
                    name = "generation 2",
                    marker = dict(color = 'rgba(168, 0, 0, 1)'),
                    text= pk2.name)
# creating trace3
trace3 =go.Scatter(
                    x = pk3.pk_number,
                    y = pk3.total,
                    mode = "markers",
                    name = "generation 3",
                    marker = dict(color = 'rgba(35, 117, 0, 1)'),
                    text= pk3.name)
# creating trace4
trace4 =go.Scatter(
                    x = pk4.pk_number,
                    y = pk4.total,
                    mode = "markers",
                    name = "generation 4",
                    marker = dict(color = 'rgba(5, 84, 133, 1)'),
                    text= pk4.name)
# creating trace5
trace5 =go.Scatter(
                    x = pk5.pk_number,
                    y = pk5.total,
                    mode = "markers",
                    name = "generation 5",
                    marker = dict(color = 'rgba(7, 181, 187, 1)'),
                    text= pk5.name)
data = [trace1, trace2, trace3, trace4, trace5]
layout = dict(title = 'Total Power vs Generation of top 100 Pokemons with gen1,gen2,gen3,gen4 and gen5',
              xaxis= dict(title= 'Pokemon Number',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Total_Power',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# prepare data frames
pk = pokemon[pokemon.generation==1].iloc[:3,:]

# import graph objects as "go"
import plotly.graph_objs as go

# create trace1 
trace1 = go.Bar(
                x = pk.name,
                y = pk.hp,
                name = "hp",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace2 
trace2 = go.Bar(
                x = pk.name,
                y = pk.attack,
                name = "attack",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace3 
trace3 = go.Bar(
                x = pk.name,
                y = pk.defense,
                name = "defense",
                marker = dict(color = 'rgba(1, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace4 
trace4 = go.Bar(
                x = pk.name,
                y = pk.speed,
                name = "speed",
                marker = dict(color = 'rgba(1, 128, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# prepare data frames
pk = pokemon[pokemon.generation==1].iloc[:3,:]

# import graph objects as "go"
import plotly.graph_objs as go

x = pk.name

trace1 = {
  'x': x,
  'y': pk.hp,
  'name': 'hp',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': pk.attack,
  'name': 'attack',
  'type': 'bar'
};
trace3 = {
  'x': x,
  'y': pk.defense,
  'name': 'defense',
  'type': 'bar'
};
trace4 = {
  'x': x,
  'y': pk.speed,
  'name': 'speed',
  'type': 'bar'
};
data = [trace1, trace2, trace3, trace4];
layout = {
  'xaxis': {'title': 'Top 3 Pokemons'},
  'barmode': 'relative',
  'title': 'Hp, Attack, Defense and Speed of top 3 Pokemons(Generation=1)'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# prepare data frames
pk = pokemon.iloc[:3,:]

# import graph objects as "go"
import plotly.graph_objs as go

# create trace1 
trace1 = go.Bar(
                x = pk.name,
                y = pk.hp,
                name = "hp",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace2 
trace2 = go.Bar(
                x = pk.name,
                y = pk.attack,
                name = "attack",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace3 
trace3 = go.Bar(
                x = pk.name,
                y = pk.defense,
                name = "defense",
                marker = dict(color = 'rgba(1, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace4 
trace4 = go.Bar(
                x = pk.name,
                y = pk.speed,
                name = "speed",
                marker = dict(color = 'rgba(1, 128, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace5
trace5 = go.Bar(
                x = pk.name,
                y = pk.sp_atk,
                name = "Special Attack",
                marker = dict(color = 'rgba(1, 128, 8, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace6
trace6 = go.Bar(
                x = pk.name,
                y = pk.sp_def,
                name = "Special Defense",
                marker = dict(color = 'rgba(128, 3, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace7 
trace7 = go.Bar(
                x = pk.name,
                y = pk.total,
                name = "Total Power",
                marker = dict(color = 'rgba(100, 255, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# prepare data frames
pk = pokemon.iloc[:3,:]

# import graph objects as "go"
import plotly.graph_objs as go

# create trace1 
trace1 = go.Bar(
                x = pk.name,
                y = pk.hp,
                name = "hp",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace2 
trace2 = go.Bar(
                x = pk.name,
                y = pk.attack,
                name = "attack",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace3 
trace3 = go.Bar(
                x = pk.name,
                y = pk.defense,
                name = "defense",
                marker = dict(color = 'rgba(1, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace4 
trace4 = go.Bar(
                x = pk.name,
                y = pk.speed,
                name = "speed",
                marker = dict(color = 'rgba(1, 128, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace5
trace5 = go.Bar(
                x = pk.name,
                y = pk.sp_atk,
                name = "Special Attack",
                marker = dict(color = 'rgba(1, 128, 8, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace6
trace6 = go.Bar(
                x = pk.name,
                y = pk.sp_def,
                name = "Special Defense",
                marker = dict(color = 'rgba(128, 3, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
# create trace7 
trace7 = go.Bar(
                x = pk.name,
                y = pk.total,
                name = "Total Power",
                marker = dict(color = 'rgba(100, 255, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = pk.type_1)
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]
layout = go.Layout(barmode = "relative")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# data preparation
pk = pokemon.total[:10]
labels = pokemon.name
# figure
fig = {
  "data": [
    {
      "values": pk,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Total Power of Pokemon",
      "hoverinfo":"label+percent+name",#name => title
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Total Power of top 10 Pokemons",
        "annotations": [
            { "font": { "size": 20}, #text size
              "showarrow": True,
              "text": "Number of Pokemons",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
# data preparation
pk = pokemon.iloc[:20,:]
pk_size  = pk.hp/3 # grafiğe göre ortalama boyu küçültüyoruz
pk_color = pk.speed
data = [
    {
        'y': pk.sp_atk,
        'x': pk.pk_number,
        'mode': 'markers',
        'marker': {
            'color': pk_color,
            'size': pk_size,
            'showscale': True
        },
        "text" :  pk.name    
    }
]
iplot(data)
# prepare data
pk1 = pokemon.total[pokemon.generation == 1]
pk2 = pokemon.total[pokemon.generation == 2]

trace1 = go.Histogram(
    x=pk1,
    opacity=0.75,
    name = "Pokemon Total Power(Generation=1)",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=pk2,
    opacity=0.75,
    name = "Pokemon Total Power(Generation=2)",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title='Pokemon Total Power Generation=1 and Generation=2',
                   xaxis=dict(title='Pokemon Total Power'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# data preparation
pk_type = pokemon.type_1
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(pk_type))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# data preparation
pk = pokemon.iloc[:,:]

trace1 = go.Box(
    y=pk.sp_atk,
    name = 'Speacial Attack of Pokemons',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace2 = go.Box(
    y=pk.sp_def,
    name = 'Speacial Defense of Pokemons',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace1, trace2]
iplot(data)
# data preparation
pk = pokemon.iloc[:,:]

trace1 = go.Box(
    x=pk.legendary,
    y=pk.sp_atk,
    name = 'Speacial Attack of Pokemons',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace2 = go.Box(
    x=pk.legendary,
    y=pk.sp_def,
    name = 'Speacial Defense of Pokemons',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)

data = [trace1, trace2]
layout = go.Layout(
    yaxis=dict(
        title='normalized moisture',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# import figure factory
import plotly.figure_factory as ff
# prepare data
pk = pokemon.iloc[:400,:]
pk_new = pk.loc[:,["total","sp_atk", "sp_def"]]
pk_new["index"] = np.arange(1,len(pk_new)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(pk_new, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
# data preparation
pk = pokemon.iloc[:100,:]

# first line plot
trace1 = go.Scatter(
    x=pk.pk_number,
    y=pk.sp_atk,
    name = "Speacial Attack",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=pk.pk_number,
    y=pk.sp_def,
    xaxis='x2',
    yaxis='y2',
    name = "Speacial Defense",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Speacial Attack and Speacial Defense'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

# data preparation
pk = pokemon.iloc[:100,:]

# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=pk.pk_number,
    y=pk.sp_atk,
    z=pk.hp,
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
# data preparation
pk = pokemon.iloc[:100,:]

trace1 = go.Scatter(
    x=pk.pk_number,
    y=pk.attack,
    name = "research"
)
trace2 = go.Scatter(
    x=pk.pk_number,
    y=pk.defense,
    xaxis='x2',
    yaxis='y2',
    name = "citations"
)
trace3 = go.Scatter(
    x=pk.pk_number,
    y=pk.speed,
    xaxis='x3',
    yaxis='y3',
    name = "income"
)
trace4 = go.Scatter(
    x=pk.pk_number,
    y=pk.total,
    xaxis='x4',
    yaxis='y4',
    name = "total_score"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
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
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Attack, Defense, Speed and Total Power of Pokemons'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)