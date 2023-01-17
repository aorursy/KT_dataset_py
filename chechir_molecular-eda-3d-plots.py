import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_absolute_error

import seaborn as sns

import lightgbm as lgb

import plotly

import plotly.graph_objs as go

import warnings



from tabulate import tabulate



warnings.filterwarnings("ignore")



plotly.offline.init_notebook_mode(connected=False)



import os

print(os.listdir("../input"))



data_dir = '../input/champs-scalar-coupling' if 'champs-scalar-coupling' in os.listdir('../input/') else '../input'
train = pd.read_csv(f'{data_dir}/train.csv')

test = pd.read_csv(f'{data_dir}/test.csv')

sub = pd.read_csv(f'{data_dir}/sample_submission.csv')

structures = pd.read_csv(f'{data_dir}/structures.csv')
print("train shape", train.shape)

print("test shape", train.shape)

print("structures shape", structures.shape)

print("sub", sub.shape)

print("train cols", list(train.columns))

print("test cols", list(test.columns))

print("structures cols", list(structures.columns))

print("structures atoms", list(np.unique(structures['atom'])))

print("")

print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")

print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")

print(f"There are {structures['atom'].nunique()} unique atoms in structures")

print(f"There are {train['type'].nunique()} unique types in train")

print(f"There are {test['type'].nunique()} unique types in test")

are_the_same_types = np.all(sorted(train['type'].unique()) == sorted(test['type'].unique()))

print(f"Are all types in train and test the same? {are_the_same_types}")
# merging

test['scalar_coupling_constant'] = np.nan

train = train.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')

train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=('_a', '_b'))

train.drop(['atom_index_a', 'atom_index_b'], axis=1, inplace=True)



test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')

test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=('_a', '_b'))

test.drop(['atom_index_a', 'atom_index_b'], axis=1, inplace=True)
print('train shape', train.shape)

print('test shape', test.shape)

train.head()
fig, axes = plt.subplots(2, 4, figsize = (15, 7), sharex=True, sharey=True)

axes = axes.flatten()

for i, type_ in enumerate(np.unique(train['type'])):

    ix = train['type'] == type_

    _ = sns.distplot(train['scalar_coupling_constant'][ix], ax=axes[i])

    axes[i].set(title=f'{type_}. Rows: {ix.sum()}')

fig.tight_layout()
fig, axes = plt.subplots(2, 4, figsize = (15, 5), sharex=True, sharey=False)

axes = axes.flatten()

subset = train[train['type'] == '1JHN'].copy().reset_index()

for i, ai1 in enumerate(subset['atom_index_1'].value_counts().index[:8]):

    ix = subset['atom_index_1'] == ai1

    _ = sns.distplot(subset['scalar_coupling_constant'][ix], ax=axes[i])

    axes[i].set(title=f'{ai1}. Rows: {ix.sum()}')

fig.tight_layout()
fig, axes = plt.subplots(2, 4, figsize = (18, 8), sharex=True, sharey=True)

axes = axes.flatten()

for i, type_ in enumerate(np.unique(train['type'])):

    ix = train['type'] == type_

    _ = sns.scatterplot(x='x_a', y='y_a', data=train.loc[ix].sample(20000), alpha=0.05, hue='scalar_coupling_constant', ax=axes[i])

    axes[i].set(title=f'{type_}. Rows: {ix.sum()}')

fig.tight_layout()
np.random.seed(10)

from plotly.graph_objs import FigureWidget

lbl = LabelEncoder()

train['type_id'] = lbl.fit_transform(train['type'])



traces = []

for i, type_ in enumerate(np.unique(train['type'])):

    ix = train['type'] == type_

    subset = train.loc[ix].sample(8000)

    trace = go.Scatter3d(

        name=type_,

        x=subset['x_a'],

        y=subset['y_a'],

        z=subset['z_a'],

        mode='markers',

        marker=dict(

            size=4,

            opacity=0.03

         )

    )

    traces.append(trace)



layout = go.Layout(

    showlegend=True,

    autosize=True,

    scene=go.Scene(),

    width=800,

    height=1000,

)



FigureWidget(data=traces, layout=layout)
subset = train[train['type'] == '1JHN'].copy().reset_index().sample(43300)



traces = go.Scatter3d(

    x=subset['x_a'],

    y=subset['y_a'],

    z=subset['z_a'],

    mode='markers',

    marker=dict(

        size=4,

        opacity=0.05,

        color=subset['scalar_coupling_constant'],

        colorscale='Viridis',

     )

)



layout = go.Layout(

    autosize=True,

    width=800,

    height=1000,

)



FigureWidget(data=[traces], layout=layout)
subset = train[train['type'] == '3JHN'].copy().reset_index().sample(43300)

traces = go.Scatter3d(

    x=subset['x_a'],

    y=subset['y_a'],

    z=subset['z_a'],

    mode='markers',

    marker=dict(

        size=4,

        opacity=0.05,

        color=subset['scalar_coupling_constant'],

        colorscale='Viridis',

     )

)



layout = go.Layout(

    autosize=True,

    width=800,

    height=1000,

)



FigureWidget(data=[traces], layout=layout)