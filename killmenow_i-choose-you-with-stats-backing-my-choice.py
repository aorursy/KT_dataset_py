import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

import collections 



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



% matplotlib inline
# lets get the dataset

poke = pd.read_csv('../input/Pokemon.csv')



#give it a little look see

poke.head()
# the '#' feature is pointless

del poke["#"]


fg = sns.FacetGrid(data= poke, hue='Generation', size = 8)

fg.map(plt.scatter, 'Attack','Defense').add_legend()

plt.axvline(x=75, color = 'red')

plt.axhline(y=75, color = 'red')
greatattack = poke['Attack']>75

greatdefense = poke['Defense']> 75

starter = poke[greatattack & greatdefense]
fg = sns.FacetGrid(data=starter, hue='Legendary', size = 8)

fg.map(plt.scatter, 'Attack','Defense').add_legend()
notlegendary = starter['Legendary'] == False

nextstarter = starter[notlegendary]
# checking the distribution of HP

dataa = [go.Bar(

            x = nextstarter['HP'].values,

            

            width = 0.5,

            marker=dict(

            color = nextstarter['HP'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    title= 'Distribution of HP',

    hovermode= 'closest',

    xaxis = dict(

        title = 'HP of Pokemon'

    ),

    showlegend= False

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barplothp')
greathp = nextstarter['HP']>75

overall = nextstarter[greathp]
fg = sns.FacetGrid(data= overall, hue='Type 1', size = 8)

fg.map(plt.scatter, 'Attack','Defense').add_legend()
dataa = [go.Bar(

            x = overall['Speed'].values,

            

            width = 0.5,

            marker=dict(

            color = overall['Speed'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    title= 'Distribution of Speed',

    hovermode= 'closest',

    xaxis = dict(

        title = 'Speed of Pokemon'

    ),

    showlegend= False

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barplotspeed')
# lets check for the mean, pretty sure 75 is good enough

np.mean(overall['Speed'].values)
# cool so 75 it is 

greatspeed = overall['Speed']>75

overall = overall[greatspeed]
# lets see how many Pokemon are left, if many are there then we will wean them out using Sp. Atk 

overall.shape
fg = sns.FacetGrid(data= overall, hue='Generation', size = 8)

fg.map(plt.scatter, 'Attack','Defense').add_legend()
# Categorize by gen

gen1 = overall['Generation'] == 1

gen2 = overall['Generation'] == 2

gen3 = overall['Generation'] == 3

gen4 = overall['Generation'] == 4

gen5 = overall['Generation'] == 5



gen1 = overall[gen1]

gen2 = overall[gen2]

gen3 = overall[gen3]

gen4 = overall[gen4]

gen5 = overall[gen5]
gen1
gen2
gen3
gen4
gen5