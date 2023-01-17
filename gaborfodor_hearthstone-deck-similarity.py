%matplotlib inline

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import datetime as dt

import re

from igraph import Graph

import plotly.graph_objects as go
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))



start = dt.datetime.now()

heroes = ['Hunter', 'Paladin', 'Mage', 'Shaman', 'Warlock', 'Druid', 'Priest', 'Warrior', 'Rogue']

deck_size = 30

card_cols = [f'card_{i}' for i in range(deck_size)]
decks = pd.read_csv('../input/hearthstone-decks/top_hearthstone_decks_20200221.csv')
card_sets = []

for _, deck in decks[card_cols].iterrows():

    card_sets.append(set(card for card in deck.values if card is not None))

decks['card_sets'] = card_sets

decks['nunique_cards'] = decks.card_sets.map(len)

decks.head()

decks.shape
df = pd.concat([

    decks.groupby('hero')[['games', 'wins']].sum(),

    decks.groupby('hero')[['dust', 'nunique_cards']].mean()

], axis=1)

df['wr'] = df.wins / df.games

df = df.sort_values(by='games', ascending=False)

df
data = [

    go.Bar(

        y=df['games'].values,

        x=df.index,

        marker=dict(

            color=df['wr'].values,

            colorscale='RdYlGn',

            showscale=True

        ),

        text=np.round(df.wr.values * 100, 1),

    )

]

layout = go.Layout(

    autosize=True,

    title='Heroes (games and winrate)',

    hovermode='closest',

    xaxis=dict(title='Heroes', ticklen=5, zeroline=False, gridwidth=0),

    yaxis=dict(title='Number of games', ticklen=5, gridwidth=2),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

fig
df = pd.concat([

    decks.groupby('type')[['games', 'wins']].sum(),

    decks.groupby('type')[['dust', 'nunique_cards']].mean()

], axis=1)

df['wr'] = df.wins / df.games

df = df.sort_values(by='games', ascending=False)

df
top_archetypes = df[:20]

data = [

    go.Bar(

        y=top_archetypes['games'].values,

        x=top_archetypes.index,

        marker=dict(

            color=top_archetypes['wr'].values,

            colorscale='RdYlGn',

            showscale=True

        ),

        text=np.round(top_archetypes.wr.values * 100, 1),

    )

]

layout = go.Layout(

    autosize=True,

    title='Most popular archetypes',

    hovermode='closest',

    xaxis=dict(title='Archetypes', ticklen=5, zeroline=False, gridwidth=0),

    yaxis=dict(title='Number of games', ticklen=5, gridwidth=2),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

fig
def jaccard(s1, s2):

    return len(s1 & s2) / len(s1 | s2)
similarity_matrix = np.zeros((len(decks), len(decks)))

edge_list = []

for i, card_set_i in enumerate(decks.card_sets.values):

    for j, card_set_j in enumerate(decks.card_sets.values):

        if i < j:

            similarity = jaccard(card_set_i, card_set_j)

            if (similarity > 0):

                similarity_matrix[i, j] = similarity

                edge = [i, j, similarity, len(card_set_i & card_set_j), card_set_i & card_set_j]

                edge_list.append(edge)
edge_df = pd.DataFrame(edge_list, columns=['i', 'j', 'similarity', 'k', 'intersection'])

edge_df.shape

edge_df.head()

edge_df.mean()
edge_df[['k', 'similarity']].hist(bins=30)

plt.suptitle('Similarity stats')

plt.plot();
similarity_threshold = 0.2

strong_edges = edge_df[edge_df['similarity'] > similarity_threshold]

strong_edges.shape
g = Graph()

g.add_vertices(len(decks))

g.vs['type'] = decks.type

g.add_edges(strong_edges[['i', 'j']].values)

g.es['weight'] = strong_edges['similarity'].values

degrees = g.degree()

g.summary()
clusters = g.clusters()

for sg in clusters.subgraphs():

    if sg.vcount() > 1:

        sg.summary()
graph_layout = g.layout('fr', weights='weight')

coords = np.array(graph_layout.coords)

deck_coords = pd.DataFrame(coords, columns=['x', 'y'])

decks['x'] = deck_coords.x

decks['y'] = deck_coords.y
data = []

for u1, u2 in strong_edges[['i', 'j']].values:

    if degrees[u1] < 20 or degrees[u2] < 20 or np.random.rand() > 0.9:

        trace = go.Scatter(

            x = [deck_coords.loc[u1, 'x'], deck_coords.loc[u2, 'x']],

            y = [deck_coords.loc[u1, 'y'], deck_coords.loc[u2, 'y']],

            mode = 'lines',

            name='edges',

            opacity=0.5,

            line=dict(color='grey', width=1))

        data.append(trace)



for hero, df in decks.groupby('hero'):

    data.append(

        go.Scatter(

            y=df['y'].values,

            x=df['x'].values,

            mode='markers',

            name=hero,

            marker=dict(sizemode='diameter',

                        sizeref=2,

                        size=df.games.values ** 0.3,

                        color='black'

                        ),

            text=df.type.values,

        )

    )

layout = go.Layout(

    autosize=True,

    title='Top decks',

    hovermode='closest',

    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

fig
data = []

for hero, df in decks.groupby('hero'):

    data.append(

        go.Scatter(

            y=df['y'].values,

            x=df['x'].values,

            mode='markers',

            name=hero,

            marker=dict(sizemode='diameter',

                        sizeref=2,

                        size=df.games.values ** 0.3,

                        ),

            text=df.type.values,

        )

    )

layout = go.Layout(

    autosize=True,

    title='Top decks',

    hovermode='closest',

    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    showlegend=True

)

fig = go.Figure(data=data, layout=layout)

fig
data = [

    go.Scatter(

        y=decks['y'].values,

        x=decks['x'].values,

        mode='markers',

        marker=dict(sizemode='diameter',

                    sizeref=2,

                    size=decks.games.values ** 0.3,

                    color=decks.wr.values,

                    colorscale='RdYlGn',

                    showscale=True

                    ),

        text=decks.type.values,

    )

]

layout = go.Layout(

    autosize=True,

    title='Top decks',

    hovermode='closest',

    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

fig
data = [

    go.Scatter(

        y=decks['y'].values,

        x=decks['x'].values,

        mode='markers',

        marker=dict(sizemode='diameter',

                    sizeref=2,

                    size=decks.games.values ** 0.3,

                    color=decks.dust.values,

                    colorscale='Viridis',

                    showscale=True

                    ),

        text=decks.type.values,

    )

]

layout = go.Layout(

    autosize=True,

    title='Top decks - Dust cost',

    hovermode='closest',

    yaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    xaxis = dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

fig
data = [

    go.Scatter(

        y=df['wr'].values,

        x=df['dust'].values,

        mode='markers',

        name=hero,

        marker=dict(sizemode='diameter',

                    sizeref=2,

                    size=df.games.values ** 0.3

                    ),

        text=decks.type.values,

    ) for hero, df in decks.groupby('hero')

]

layout = go.Layout(

    autosize=True,

    title='Top decks - Dust vs winrate',

    hovermode='closest',

    xaxis=dict(title='Dust', ticklen=5, zeroline=False, gridwidth=0),

    yaxis=dict(title='Win rate', ticklen=5, gridwidth=2),

    showlegend=True

)

fig = go.Figure(data=data, layout=layout)

fig
decks['expected_star_per_turn'] = (3 * (2 * decks.wr - 1) + decks.wr * decks.wr * decks.wr) / 3

decks['star_speed'] = 60 * decks.expected_star_per_turn / decks.duration
decks.sort_values(by='star_speed', ascending=False)[[

    'type', 'dust', 'wr', 'games', 'duration', 'expected_star_per_turn', 'star_speed']].head(10)

decks.sort_values(by='games', ascending=False)[[

    'type', 'dust', 'wr', 'games', 'duration', 'expected_star_per_turn', 'star_speed']].head(10)
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))