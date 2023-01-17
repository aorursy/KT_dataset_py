# Realizando Importações

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
#Importando arquivo do GitHub

url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv"
pokedex = pd.read_csv(url, sep=',')
pokedex.head()
# Classificação das Variáveis

resposta = [["#","Quantitativa Continua"],["Name","Qualitativa Nominal"],["Type 1","Qualitativa Nominal"],
            ["Type 2","Qualitativa Nominal"],["Total","Quantitativa Discreta"],["HP","Quantitativa Nominal"],
            ["Attack","Quantitativa Nominal"],["Defense","Quantitativa Nominal"],["Sp. Atk","Quantitativa Nominal"],
            ["Sp. Def","Quantitativa Nominal"],["Speed","Quantitativa Nominal"],["Stage","Quantitativa Ordinal"],
            ["Legendary","Qualitativa Ordinal"]]

variaveis = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
variaveis
def stats_by_type(type):
    data, stats_names = [], ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    particular_type = pokedex[pokedex['Type 1'] == type].reset_index(drop=True)
    for stat in stats_names:
        stat_line = go.Scatter(
            x=particular_type['Name'],
            y=particular_type[stat],
            name=stat,
            line=dict(
                width=3
            ))

        data.append(stat_line)

    layout = go.Layout(
        title="Stats of every Pokemon of {} type".format(type),
        xaxis=dict(
            title="{} Pokemon".format(type)
        ),
        yaxis=dict(
            title="Values"
        ))

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='type_stats')

stats_by_type('Normal')
c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360)]

types = (pokedex.groupby(['Type 1'])['#'].count())
types_name = list(types.keys())

data = go.Bar(
    x=types_name,
    y=types.values,
    marker=dict(
        color=np.random.randn(123), opacity=0.8),
    name="{}".format(types_name))

layout = go.Layout(
    title= 'Visualizando o número de pokemon por tipo em toda a geração',
    xaxis=dict(
        title='Type'
    ),
    yaxis=dict(
        title='Quantidade de Pokémons'
    ))

fig = go.Figure(data=[data], layout=layout)
iplot(fig, filename='Types')
x = pokedex['Sp. Atk'].values
data = [go.Histogram(x=x)]

layout = dict(
    title='Distribuição das Estatísticas Especiais de ataque entre os Pokémons no geral',
            autosize= True,bargap= 0.015,hovermode= 'x',
    xaxis=dict(
        autorange= True,
        zeroline= False,
        title='Estatísticas'),
    yaxis= dict(
        autorange= True,
        showticklabels= True,
        title='Valores'))

fig1 = dict(data=data, layout=layout)
iplot(fig1)
x = pokedex[pokedex["Name"] == "Charizard"]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself',
     line =  dict(
            color = 'orange'))]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200])),
  showlegend = False,
  title = 'Estatística de Distribuição do Charizard')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
x = pokedex[pokedex["Name"] == "Blastoise"]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself',
     line =  dict(
            color = 'light blue'))]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200])),
  showlegend = False,
  title = 'Estatística de Distribuição do Blastoise')

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename = "Single Pokemon stats")
a = pokedex[pokedex["Name"] == "Blastoise"]
b = pokedex[pokedex["Name"] == "Charizard"]

data = [
    go.Scatterpolar(
        name = a.Name.values[0],
        r = [a['HP'].values[0],a['Attack'].values[0],a['Defense'].values[0],a['Sp. Atk'].values[0],a['Sp. Def'].values[0],a['Speed'].values[0],a["HP"].values[0]],
        theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
        fill = 'toself',
        line =  dict(
                color = 'light blue')),
    
    go.Scatterpolar(
            name = b.Name.values[0],
            r = [b['HP'].values[0],b['Attack'].values[0],b['Defense'].values[0],b['Sp. Atk'].values[0],b['Sp. Def'].values[0],b['Speed'].values[0],b["HP"].values[0]],
            theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
            fill = 'toself',
            line =  dict(
                color = 'orange'))]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 200])),
  showlegend = True,
  title = 'Blastoise vs. Charizard Comparação Estatística')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
box0 = go.Box(y=pokedex["HP"],name="HP")
box1 = go.Box(y=pokedex["Attack"],name="Attack")
box2 = go.Box(y=pokedex["Defense"],name="Defense")
box3 = go.Box(y=pokedex["Sp. Atk"],name="Sp. Atk")
box4 = go.Box(y=pokedex["Sp. Def"],name="Sp. Def")
box5 = go.Box(y=pokedex["Speed"],name="Speed")

layout = go.Layout(
        title='Boxplots de todas as estatísticas',
        xaxis=dict(title='Estatísticas'),
        yaxis=dict(title='Valores'))

data = [box0, box1, box2,box3, box4, box5]
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='type_stats')
fig = {'data': [{'type' : 'violin',
                 'x' : pokedex['Legendary'].astype(str),
                 'y' : pokedex['Attack'],
                 'legendgroup' : 'Attack',
                 'scalegroup' : 'Attack',
                 'name' : 'Ataque',
                 'side' : 'negative',
                 'box' : {'visible' : True},
                 'points' : 'all',
                 'pointpos' : -1.15,
                 'jitter' : 0.1,
                 'scalemode' : 'probability', # 'count'
                 'meanline' : {'visible' : True},
                 'line' : {'color' : 'blue'},
                 'marker' : {'line' : {'width': 0,'color' : '#000000'}},
                 'span' : [0],
                 'text' : pokedex['Name']},
                
                {'type' : 'violin',
                 'x' : pokedex['Legendary'].astype(str),
                 'y' : pokedex['Defense'],
                 'legendgroup' : 'Defense',
                 'scalegroup' : 'Defense',
                 'name' : 'Defesa',
                 'side' : 'positive',
                 'box' : {'visible' : True},
                 'points' : 'all',
                 'pointpos' : 1.15,
                 'jitter' : 0.1,
                 'scalemode' : 'probability', #'count'
                 'meanline' : {'visible': True},
                 'line' : {'color' : 'green'},
                 'marker' : {'line' : {'width' : 0,'color' : '#000000'}},
                 'span' : [1],
                 'text' : pokedex['Name']}],
       'layout' : {'title' : 'Ataque Pokemon agrupado por Lendário',
                   'xaxis' : {'title' : 'Lendário'},
                   'yaxis' : {'zeroline' : False,
                              'title' : 'Ataque / Defesa'},
                   'violingap' : 0,
                   'violinmode' : 'overlay'}}

iplot(fig, validate=False)