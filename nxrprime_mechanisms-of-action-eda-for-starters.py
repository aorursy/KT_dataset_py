import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

df = pd.read_csv('../input/lish-moa/train_features.csv')

plt.style.use('seaborn-darkgrid')
df
for i in df.isnull().any():

    if i == True:

        print("MISSING VALUE")
targs = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

targs.head()
fig, axes = plt.subplots(8,2,figsize=(14, 30), dpi=100)

for i in range(0, 16):

    df[f"g-{i}"].plot(ax=axes[i%8][i//8], alpha=0.8, label='Feature', color='tab:blue')

    df[f"g-{i}"].rolling(window=4).mean().plot(ax=axes[i%8][i//8], alpha=0.8, label='Rolling mean', color='tab:orange')

    axes[i%8][i//8].legend();

    axes[i%8][i//8].set_title('Feature {}'.format(i), fontsize=13);

    plt.subplots_adjust(hspace=0.45)

from sklearn.decomposition import PCA

df = df.drop(["sig_id","cp_type","cp_time","cp_dose"],axis=1)

pca = PCA(n_components=6)

pca.fit(df)

pca_samples = pca.transform(df)

ps = pd.DataFrame(pca_samples)

ps.head()
fig, axes = plt.subplots(3,2,figsize=(7, 15), dpi=100)

for i in range(0, 6):

    ps[i].plot(ax=axes[i%3][i//3], alpha=0.8, label='Feature', color='tab:blue')

    ps[i].rolling(window=4).mean().plot(ax=axes[i%3][i//3], alpha=0.8, label='Rolling mean', color='tab:orange')

    axes[i%3][i//3].legend();

    axes[i%3][i//3].set_title('Feature {}'.format(i), fontsize=13);

    plt.subplots_adjust(hspace=0.45)

fig = plt.figure(figsize=(8, 8))

sns.heatmap(ps.corr(), annot=True, cmap=plt.cm.magma);
import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

import plotly.offline as py

py.init_notebook_mode(connected=True)

data = [

    go.Heatmap(

        z= ps.corr().values,

        x=ps.columns.values,

        y=ps.columns.values,

        colorscale='Viridis',

        reversescale = False,

        opacity = 1.0 )

]



layout = go.Layout(

    title='Pearson Correlation of Integer-type features',

    xaxis = dict(ticks='', nticks=36),

    yaxis = dict(ticks='' ),

    width = 900, height = 700)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
fig, axes = plt.subplots(8,2,figsize=(14, 30), dpi=100)

for i in range(0, 16):

    df[f"g-{i}"].plot(ax=axes[i%8][i//8], alpha=0.8, label='Feature', color='tab:blue')

    df[f"g-{i}"].rolling(window=4).std().plot(ax=axes[i%8][i//8], alpha=0.8, label='Rolling mean', color='tab:orange')

    axes[i%8][i//8].legend();

    axes[i%8][i//8].set_title('Feature {}'.format(i), fontsize=13);

    plt.subplots_adjust(hspace=0.45)

df = pd.read_csv('../input/lish-moa/train_features.csv')



sns.countplot(df['cp_type'])
print("WORK IN PROGRESS")