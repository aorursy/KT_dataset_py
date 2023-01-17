#Importing Libraries
import pandas as pd
import numpy as np
#Plotly Offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go

init_notebook_mode(connected=True)

#Checking file list (Because I was a little bit confused)
import os
print(os.listdir("../input"))
#Importing File
file_path = '../input/aula-3-exercicio2/'
file_name = 'Pokemon.csv'
df = pd.read_csv(file_path + file_name,delimiter = ',')

#Reading Dataset
df.head(10)
#Deixe claro qual o dataset escolhido e as variáveis.

resposta = [["Name", "Qualitativa Nominal"],["Type 1", "Qualitativa Nominal"],["Total", "Quantitativa Continua"],["HP", "Quantitativa Continua"],["Attack", "Quantitativa Continua"],["Defense", "Quantitativa Continua"],["Sp. Atk", "Quantitativa Continua"],["Sp. Def", "Quantitativa Continua"],["Speed", "Quantitativa Continua"]]
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
#Um gráfico de linha
y = df['Attack']
y1 = df['Defense']

# Create a trace
Attack  = go.Scatter(y = y,name = 'Attack')
Defense = go.Scatter(y = y1,name = 'Defense')

data = [Attack,Defense]

offline.iplot(data, filename='basic-line')
#Um gráfico de barras
#Pokemon by type

y= df.groupby('Type 1')['Name'].count().sort_values()
x = df['Type 1'].unique()

data = [go.Bar(x= x,y= y)]

layout = go.Layout(
    title='Pokemon by Type',
)

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='text-hover-bar')
#Um histograma
#Spped A histogram
x = df['Total']
data = [go.Histogram(x=x)]
layout = go.Layout(title='Pokemon Total Score Histogram')

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename='basic histogram')
# Um gráfico radar
#Bulbasaur Radar Chart

mask = df['Name'] == 'Bulbasaur'
x0 = df[mask]
x1 = x0[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

data = [go.Scatterpolar(
  r = x1.values.ravel(),
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],
  fill = 'toself'
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100],
    ),
  ),
  title = 'Bulbasaur Radar Chart',
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename = "radar/basic")
#Um gráfico de caixa
y0 = df['HP']
y1 = df['Attack']
y2 = df['Defense']
y3 = df['Speed']

trace0 = go.Box(y=y0,name = 'HP')
trace1 = go.Box(y=y1,name = 'Attack')
trace2 = go.Box(y=y2,name = 'Defense')
trace3 = go.Box(y=y3,name = 'Speed')

data = [trace0, trace1,trace2,trace3]
offline.iplot(data)
#Um gráfico de violino
import plotly.figure_factory as ff
y = df['Defense'].values.tolist()

fig = ff.create_violin(y, title='Defense Violin Plot', colors='#604d9e')
offline.iplot(fig, filename='Violin Visual')