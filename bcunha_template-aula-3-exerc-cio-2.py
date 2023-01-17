# Importing libraries and resources
import pandas as pd

import numpy as np

import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.figure_factory as ff
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
plotly.tools.set_credentials_file(username = 'bscunha', api_key = 'QoHRVgmNUWhee8y9LinC')
# Loading the Pokémon csv file to pandas dataframe
pokemon = pd.read_csv('https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/Pokemon.csv')
pokemon.rename(columns = {'Type 1': 'Type', 'Type 2': 'Subtype', 'Sp. Atk': 'SpAtk', 'Sp. Def': 'SpDef'}, inplace = True)
pokemon.head() # preview of the first lines of the dataframe
types = (pokemon.groupby('Type')['Total'].count())
types_name = types.keys().tolist()
data_types = go.Bar(name = 'Type', x = types_name, y = types.values) # text=types, textposition = 'auto'

subtypes = (pokemon.groupby('Subtype')['Total'].count())
subtypes_name = subtypes.keys().tolist()
data_subtypes = go.Bar(name = 'Subtype', x = subtypes_name, y = subtypes.values)

layout = go.Layout(
    barmode = 'group', title = 'Number of Pokémons by type',
    xaxis = dict(title = 'Type'), yaxis = dict(title = 'Number of Pokemon')
)

fig = go.Figure(data = [data_types, data_subtypes], layout = layout)
py.iplot(fig, filename = 'Number of Pokémons by type')
pokemon.Attack.describe()
# Just proving the theory that at least 103 of the 150 Pokémons must be from 45 to 99 points of attack
pokemon[(pokemon.Attack >= 45) & (pokemon.Attack < 100)].count().tolist()[0]
attack_distplot = ff.create_distplot([pokemon.Attack], ['Attack'], bin_size = 5)
attack_distplot['layout'].update(title = 'Attack Distplot')
py.iplot(attack_distplot, filename = 'Attack Distplot')
# Finding the strongest and the weakest Pokémons, based on 'Total' attribute
# Colors took from <http://pokepalettes.com/>
sort_by_total = pokemon.sort_values(by = 'Total', ascending = False)
sort_by_total.head(1).append(sort_by_total.tail(1))
Mewtwo = pokemon[pokemon.Name == 'Mewtwo']
Weedle = pokemon[pokemon.Name == 'Weedle']
attributes = ['HP','Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def']

data = [
    go.Scatterpolar(
        name = 'Mewtwo',
        r = [Mewtwo.HP, Mewtwo.Attack, Mewtwo.Defense, Mewtwo.Speed, Mewtwo.SpAtk, Mewtwo.SpDef],
        theta = attributes, fill = 'toself', marker = dict(color = '#837b9c')
    ),
    go.Scatterpolar(
        name = 'Weedle',
        r = [Weedle.HP, Weedle.Attack, Weedle.Defense, Weedle.Speed, Weedle.SpAtk, Weedle.SpDef],
        theta = attributes, fill = 'toself', marker = dict(color = '#cd7310')
    )
]

layout = go.Layout(
    polar = dict(radialaxis = dict(visible = True, range = [0, 160])),
    showlegend = True, title = 'Weedle vs Mewtwo Stats Comparison'
)

fig = go.Figure(data = data, layout = layout)

fig.layout.images = [
    dict(source = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/150.png',
    xref = 'paper', yref = 'paper', x = 0.95, y = 0.3, sizex = 0.4, sizey = 0.4, xanchor = 'center', yanchor = 'bottom'),

    dict(source='https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/13.png',
    xref = 'paper', yref = 'paper', x = 0.05, y = 0.3, sizex = 0.4, sizey = 0.4, xanchor = 'center', yanchor = 'bottom')
]

py.iplot(fig, filename = 'Weedle vs Mewtwo Stats Comparison')
pokemon.corr()
pokemon.corr().abs().where(np.triu(np.ones(pokemon.corr().shape), k=1).astype(np.bool)).stack().nlargest(2)
data = go.Scatter(
    x = pokemon.SpDef, y = pokemon.Total,
    mode = 'markers', text = pokemon.Name,
    marker = dict(size = 10, color = pokemon.SpAtk, showscale = True)
)

layout = go.Layout(
    title = 'Scatter plot of Total by the Sp. Def, colored on Sp. Atk',
    xaxis = dict(title = 'Sp. Def'), yaxis = dict(title = 'Total'),
    showlegend = False
)

correlation_stats = go.Figure(data = [data], layout = layout)

py.iplot(correlation_stats)
hp = go.Box(y = pokemon.HP, name = 'HP')
attack = go.Box(y = pokemon.Attack, name = 'Attack')
defense = go.Box(y = pokemon.Defense, name = 'Defense')
sp_atk = go.Box(y = pokemon.SpAtk, name = 'Sp. Atk')
sp_def = go.Box(y = pokemon.SpDef, name = 'Sp. Def')
speed = go.Box(y = pokemon.Speed, name = 'Speed')

layout = go.Layout(
    title = 'Boxplots of all characteristics',
    xaxis = dict(title = 'Characteristic'), yaxis = dict(title = 'Value')
)

fig = go.Figure(data = [hp, attack, defense, sp_atk, sp_def, speed], layout = layout)

py.iplot(fig, filename = 'characteristics_boxplot')
data = []
for i in range(5, 11):
    trace = {
        'type': 'violin',
        'x': max(pokemon.iloc[:,i]),
        'y': pokemon.iloc[:,i],
        'name': list(pokemon.columns)[i],
        'box': {'visible': True},
        'meanline': {'visible': True}
    }
    data.append(trace)
        
fig = {
    'data': data,
    'layout' : {
        'title': 'Violin plot of all stats',
        'yaxis': {'zeroline': False}
    }
}

py.iplot(fig, filename = 'violin', validate = False)
