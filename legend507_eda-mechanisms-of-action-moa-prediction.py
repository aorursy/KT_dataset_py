import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import plotly.express as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



pd.set_option("display.max_columns", 999)
dir = '/kaggle/input/lish-moa/'

df_train = pd.read_csv(dir + 'train_features.csv')

df_test = pd.read_csv(dir + 'test_features.csv')
df_train.head(10)
df_test.head(10)
df_train_scored = pd.read_csv(dir + 'train_targets_nonscored.csv')

df_train_noscored = pd.read_csv(dir + 'train_targets_scored.csv')
df_train_scored.head(10)
# What MoAs does scored csv contain? And the frequency for a MoA to happen.



tmp = df_train_scored.drop(['sig_id'], axis = 1

                          ).sum(axis=0).sort_values().reset_index()

tmp.columns = ['MoA', '# of 1s']

fig = px.bar(

    tmp.tail(80),

    x = '# of 1s',

    y = 'MoA',

    orientation = 'h',

    height = 1000, 

    width = 800,

)

fig.show()
df_train_noscored.head(10)
tmp = df_train_noscored.drop(['sig_id'], axis = 1

                          ).sum(axis=0).sort_values().reset_index()

tmp.columns = ['MoA', '# of 1s']

fig = px.bar(

    tmp.tail(80),

    x = '# of 1s',

    y = 'MoA',

    orientation = 'h',

    height = 1000, 

    width = 800,

)

fig.show()
# This cell shows all columns with the same name in the 2 target files.



set(df_train_scored.columns) & set(df_train_noscored.columns)