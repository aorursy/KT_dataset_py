import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline





import seaborn as sns

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()



df = pd.read_csv('../input/2015.csv')

df.head(3)
corrmat = df.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
g = sns.stripplot(x="Region", y="Happiness Rank", data=df, jitter=True)

plt.xticks(rotation=90)
data = dict(type = 'choropleth', 

           locations = df['Country'],

           locationmode = 'country names',

           z = df['Happiness Rank'], 

           text = df['Country'],

           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)