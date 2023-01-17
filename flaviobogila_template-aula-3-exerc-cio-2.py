%matplotlib inline
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
df = pd.read_csv("https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv")
df.head(5)
resposta = [
    ["Name", "Qualitativa Nominal"],["Type 1","Qualitativa Nominal"],["Type 2","Qualitativa Nominal"],
    ["Total","Quantitativa Discreta"],["HP","Quantitativa Discreta"], ["Attack","Quantitativa Discreta"],
    ["Defense","Quantitativa Discreta"],["Sp. Atk","Quantitativa Discreta"], ["Sp. Def","Quantitativa Discreta"],
    ["Speed","Quantitativa Discreta"],["Legendary","Quantitativa Discreta"],
]
pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
df1 = df[df.Legendary == True]

trace1 = go.Scatter( x=df1['Name'], y=df1['Attack'], name='Attack' )
trace2 = go.Scatter( x=df1['Name'], y=df1['Defense'], name='Defense' )
trace3 = go.Scatter( x=df1['Name'], y=df1['HP'], name='HP' )
trace4 = go.Scatter( x=df1['Name'], y=df1['Speed'], name='Speed')

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(title='As habilitadades dos Pokemons legendarios' )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
types1 = df.groupby('Type 1').size()
types2 = df.groupby('Type 2').size()

trace1 = go.Bar( x=types1.keys(), y=types1.values, name='Type 1' )
trace2 = go.Bar( x=types2.keys(), y=types2.values, name='Type 2' )

data = [trace1, trace2]
layout = go.Layout( title='A quantidade de Pokemons por tipo')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
py.iplot([go.Histogram(y=df['Total'])])
skeels = df.keys().tolist()[5:11]
pokemons = df[(df.Name == 'Bulbasaur')  | (df.Name == 'Charmander') 
              | (df.Name == 'Pikachu') | (df.Name == 'Squirtle')].values.tolist()
colors = {0 :'#0B610B', 1 : '#DF0101', 2 : '#0080FF', 3: '#FFFF00'}
data = []
for i, c in colors.items():
    trace = go.Scatterpolar(
        r = pokemons[i][5:11], theta = skeels, fill = 'toself', name = pokemons[i][1], marker=dict( color=c)
    )
    data.append(trace)

layout = go.Layout( polar = dict( radialaxis = dict( visible = True, range = [0, 100] )),  showlegend = True)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace0 = go.Box( y=df['Attack'], name = 'Attack' )
trace1 = go.Box( y=df['Defense'], name = 'Defense')
py.iplot([trace0, trace1])
data = []
for p in df[0:0].keys().tolist()[5:11]:
    trace = {
        "type": 'violin', "y": df[p], "name": p,
        "box": { "visible": True },
        "meanline": { "visible": True },
    }
    data.append(trace)

py.iplot(data, filename = 'violin/basic', validate = False)