%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

url = "https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv"
df_pokemon = pd.read_csv(url)
df_pokemon.head()
legendary = (df_pokemon[df_pokemon['Legendary'] == True].groupby(by=['Type 1','Legendary']).sum()['HP']).reset_index().rename(columns={'index':'Type 1','HP':'Legendarys HP' })
legendary.drop('Legendary', axis = 1,inplace = True)
comuns = (df_pokemon[df_pokemon['Legendary'] == False].groupby(by=['Type 1','Legendary']).sum()['HP']).reset_index().rename(columns={'index':'Type 1','HP':'Commons HP'})
comuns.drop('Legendary', axis = 1,inplace = True)
pokedex = pd.merge(comuns,legendary ,how='left', on='Type 1')
pokedex.fillna(0, inplace=True)
pokedex
trace1 = go.Scatter(
    x = pokedex['Type 1'],
    y = pokedex['Legendarys HP'],
    mode = 'lines',
    name = 'Lendarios',
)


trace2 = go.Scatter(
    x = pokedex['Type 1'],
    y = pokedex['Commons HP'],
    mode = 'lines',
    name = 'Comuns',
)

data = [trace1, trace2]

plotly.offline.iplot(data, filename='basic-line')
df_name =  df_pokemon[( ( df_pokemon['Name'] == 'Bulbasaur') | (df_pokemon['Name'] == 'Charmander')  | (df_pokemon['Name'] == 'Squirtle')  | (df_pokemon['Name'] == 'Pikachu') ) ] 
df_barras = (df_name[['Name', 'Speed','Attack','Defense']].groupby(by=['Name'])).mean().reset_index().rename(columns={'index':'Name'})
df_barras
trace1 = go.Bar(
    x = df_barras['Name'] ,
    y = df_barras['Attack'],
    text = df_barras['Attack'] ,
    name ='Attack',
    textposition = 'auto',
    marker=dict(
        color='blue',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
   
)

trace2 = go.Bar(
    x = df_barras['Name'] ,
    y = df_barras['Defense'],
    text = df_barras['Defense'] ,
    name ='Defesa',
    textposition = 'auto',
    marker=dict(
        color='green',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
 
)

trace3 = go.Bar(
    x = df_barras['Name'] ,
    y = df_barras['Speed'],
    text = df_name['Speed'] ,
    name ='Velocidade',
    textposition = 'auto',
    marker=dict(
        color='red',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),

)


data = [trace1,trace2,trace3]
plotly.offline.iplot(data, filename='grouped-bar-direct-labels')

x = df_pokemon['Speed'].values
data = [go.Histogram(x=x)]



layout = go.Layout(
    xaxis=dict(
        title='',
        zeroline=False,
        showgrid=False,
         showline=False,
    )
)

fig = go.Figure(data=data, layout=layout)


plotly.offline.iplot(fig)
Ground_HP = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['HP']
Ground_Attack = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Attack']
Ground_Defense = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Defense']
Ground_SpAtk = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Sp. Atk']
Ground_SpDef = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Sp. Def']
Ground_Speed = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Speed']
Ground_Stage = df_pokemon[df_pokemon['Type 1'] == 'Ground'].mean()['Stage']

data = [
    go.Scatterpolar(
      r = [Ground_HP, Ground_Attack, Ground_Defense, Ground_SpAtk, Ground_SpDef, Ground_Speed, Ground_Speed , Ground_Stage],
      theta = ['HP','Attack','Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Stage'],
      fill = 'toself',
      name = 'Ground Pokemons'
    )

]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100]
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename = "radar/multiple")
plotly.offline.iplot(fig, filename = "radar")
trace0 = go.Box( 
  y= df_pokemon[df_pokemon['Legendary'] == False]['Sp. Atk'] ,
    name='Sp. Atk',
     boxmean='sd',
    marker=dict(
        color='rgb(8, 81, 156)'
        #color='#1E88E5'
    )
)
trace1 = go.Box(
    y=df_pokemon[df_pokemon['Legendary'] == False]['Sp. Def'],  
    name='Sp. Def',
     boxmean='sd',
    marker=dict(
        color='green'
    )
)

data = [trace0, trace1]
layout = go.Layout(
   yaxis=dict(
        title='',
        zeroline=False,
        showgrid=False,
         showline=True,
    ),   
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename = "box plot")
df_type1 =  df_pokemon[( ( df_pokemon['Type 1'] == 'Ground') | (df_pokemon['Type 1'] == 'Rock')  | (df_pokemon['Type 1'] == 'Normal')  | (df_pokemon['Type 1'] == 'Grass') ) ] 

#for i in range(0,len(pd.unique(df['day']))):
data = []


trace = {
        "type": 'violin',
        "x": 'Ground',
        "y": df_type1[ df_type1['Type 1'] == 'Ground']['Defense'],
        "name": 'Ground',
        "fillcolor": '#D2691E',
         "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

trace = {
        "type": 'violin',
        "x": 'Rock',
        "y": df_type1[ df_type1['Type 1'] == 'Rock']['Defense'],
        "name": 'Rock',
        "fillcolor": '#C0C0C0',
     "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)

trace = {
        "type": 'violin',
        "x": 'Normal',
        "y": df_type1[ df_type1['Type 1'] == 'Normal']['Defense'],
        "name": 'Normal',
         "fillcolor": '#00ccff',
     "line": {
            "color": 'black'
        },
        "box": {
            "visible": True
        },
        "meanline": {
            "visible": True
        }
    }
data.append(trace)
       
fig = {
    "data": data,
    "layout" : {
        "title": "",
        "yaxis": {
            "zeroline": False,
        }
    }
}

plotly.offline.iplot(fig, filename='violin/multiple', validate = False)
