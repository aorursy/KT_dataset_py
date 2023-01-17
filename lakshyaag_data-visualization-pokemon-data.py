# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objs as go
from IPython.display import HTML
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon.head()
# Defining colors for graphs 
colors = {
    "Bug": "#A6B91A",
    "Dark": "#705746",
    "Dragon": "#6F35FC",
    "Electric": "#F7D02C",
    "Fairy": "#D685AD",
    "Fighting": "#C22E28",
    "Fire": "#EE8130",
    "Flying": "#A98FF3",
    "Ghost": "#735797",
    "Grass": "#7AC74C",
    "Ground": "#E2BF65",
    "Ice": "#96D9D6",
    "Normal": "#A8A77A",
    "Poison": "#A33EA1",
    "Psychic": "#F95587",
    "Rock": "#B6A136",
    "Steel": "#B7B7CE",
    "Water": "#6390F0",
}
# HP distplot
hp_distplot = ff.create_distplot([pokemon.HP], ['HP'], bin_size=5)
iplot(hp_distplot, filename='HP Distplot')
# Attack / Defense distplot
attack_defense_distplot = ff.create_distplot([pokemon.Attack, pokemon.Defense], ['Attack', 'Defense'], bin_size=5)
iplot(attack_defense_distplot, filename='Attack/Defense Distplot')
# Visualizing single Pokemon statistics


def polar_pokemon_stats(pkmn_name):
    pkmn = pokemon[pokemon.Name == pkmn_name]
    obj = go.Scatterpolar(
        r=[
            pkmn['HP'].values[0],
            pkmn['Attack'].values[0],
            pkmn['Defense'].values[0],
            pkmn['Sp. Atk'].values[0],
            pkmn['Sp. Def'].values[0],
            pkmn['Speed'].values[0],
            pkmn['HP'].values[0]
        ],
        theta=[
            'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'HP'
        ],
        fill='toself',
        marker=dict(
            color=colors[pkmn['Type 1'].values[0]]
        ),
        name=pkmn['Name'].values[0]
    )

    return obj


def plot_single_pokemon(name):
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 250]
            )
        ),
        showlegend=False,
        title="Stats of {}".format(name)
    )

    pokemon_figure = go.Figure(data=[polar_pokemon_stats(name)], layout=layout)
    iplot(pokemon_figure, filename='Single Pokemon')

name = 'Charmander'
plot_single_pokemon(name)
# Comparing stats of 2 different pokemons
pkmn_1_name = 'Kyogre'
pkmn_2_name = 'Entei'

layout = go.Layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 250]
        )
    ),
    showlegend=True,
    title="{} vs. {}".format(pkmn_1_name, pkmn_2_name)
)

pkmn_1 = polar_pokemon_stats(pkmn_1_name)
pkmn_2 = polar_pokemon_stats(pkmn_2_name)

compare_2_pokemon = go.Figure(data=[pkmn_1, pkmn_2], layout=layout)
iplot(compare_2_pokemon, filename='Compare 2 Pokemon')
# Correlation of 2 stats based on a third stat (default = Speed)


def get_correlation_stats(stat1, stat2, stat3='Speed'):
    data = go.Scatter(
        x=pokemon[stat1],
        y=pokemon[stat2],
        mode='markers',
        marker=dict(
            size=10,
            color=pokemon[stat3],
            colorscale='Viridis',
            showscale=True
        ),
        text=pokemon['Name']
    )

    layout = go.Layout(
        xaxis=dict(
            title=stat1
        ),
        yaxis=dict(
            title=stat2
        ),
        showlegend=False,
        title="Scatter plot of {} and {}, colored on {}".format(stat1, stat2, stat3)
    )

    correlation_stats = go.Figure(data=[data], layout=layout)
    return correlation_stats


iplot(get_correlation_stats('HP', 'Defense'), filename='Stats Correlation')
# Attack vs. Defense of pokemon over generations, sized by HP


def attack_vs_def(type):
    type_data = pokemon[pokemon['Type 1'] == type]
    data = []

    for i in range(1, 7):
        gen = type_data[type_data['Generation'] == i]
        trace = go.Scatter(
            x=gen['Attack'],
            y=gen['Defense'],
            mode='markers',
            marker=dict(
                symbol='circle',
                sizemode='area',
                size=gen['HP'],
                sizeref=2. * max(gen['HP']) / (2000),
                line=dict(
                    width=2
                ),
            ),
            name='Generation {}'.format(i),
            text=type_data['Name']
        )
        data.append(trace)

    layout = go.Layout(
        showlegend=True,
        xaxis=dict(
            title="Attack"
        ),
        yaxis=dict(
            title="Defense"
        ),
        title="Attack vs. Defense of {} pokemon over generations, sized by HP".format(type)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


iplot(attack_vs_def('Electric'), filename='Fire Over Generation')
# Visualizing number of pokemon per type across all generation.

types = (pokemon.groupby(['Type 1'])['#'].count())
types_name = list(types.keys())

data = go.Bar(
    x=types_name,
    y=types.values,
    marker=dict(
        color=list(colors.values())
    ),
    name="{}".format(types_name)
)

layout = go.Layout(
    title='Types',
    xaxis=dict(
        title='Type'
    ),
    yaxis=dict(
        title='Number of Pokemon'
    )
)

fig = go.Figure(data=[data], layout=layout)
iplot(fig, filename='Types')

# Visualizing the trend of stats by type and generation.
def stats_by(classifier):
    data = []
    stats_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stats = pokemon.groupby(classifier)[stats_names].mean().reset_index()
    for stat in stats_names:
        stat_line = go.Scatter(
            x=stats[classifier],
            y=stats[stat],
            name=stat,
            line=dict(
                width=3,
            ),
        )

        data.append(stat_line)

    layout = go.Layout(
        title='Trend of stats by {}'.format(classifier),
        xaxis=dict(title=classifier),
        yaxis=dict(title='Values')
    )

    trend = go.Figure(data=data, layout=layout)
    iplot(trend, filename='trend')


stats_by('Generation')
stats_by('Type 1')

# Visualzing stats of pokemon per type


def stats_by_type(type):
    data = []
    stats_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    particular_type = pokemon[pokemon['Type 1'] == type].reset_index(drop=True)
    for stat in stats_names:
        stat_line = go.Scatter(
            x=particular_type['Name'],
            y=particular_type[stat],
            name=stat,
            line=dict(
                width=3
            )
        )

        data.append(stat_line)

    layout = go.Layout(
        title="Stats of every Pokemon of {} type".format(type),
        xaxis=dict(
            title="{} Pokemon".format(type)
        ),
        yaxis=dict(
            title="Values"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='type_stats')


stats_by_type('Normal')