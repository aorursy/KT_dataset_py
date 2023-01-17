import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

import plotly.plotly as py

import plotly.graph_objs as go

from collections import Counter

from math import log
def df_group(df, column):

    name_list = []

    count_list = []

    ele_col_list = pd.unique(df[column])

    ele_col_list.sort()

    for i in ele_col_list:

        discriminated = df.loc[data[column] == i]

        count = discriminated.size

        name_list.append(i)

        count_list.append(count)

        

    return name_list, count_list



def set_axis_style(ax, labels, name):

    ax.get_xaxis().set_tick_params(direction='out')

    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks(np.arange(1, len(labels) + 1))

    ax.set_xticklabels(labels)

    ax.set_xlim(0.25, len(labels) + 0.75)

    ax.set_xlabel(name)
data = pd.read_csv('/kaggle/input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')
labels, counts = df_group(data, 'Gender')

print(labels)

fig, axes = plt.subplots()

name = 'Valor X Genero'

#data.loc[data['Gender'] == 'M', 'Purchase']

#axes.violinplot([data.loc[data['Gender'] == 'M', 'Purchase'].values, [data.loc[data['Gender'] == 'F', 'Purchase'].values]], labels=labels, showmeans=True, showextrema=True, showmedians=True)

axes.violinplot([data.loc[data['Gender'] == 'F', 'Purchase'].values, data.loc[data['Gender'] == 'M', 'Purchase'].values],showmeans=True, showextrema=True)

set_axis_style(axes, labels, name)

counts = data['Product_ID'].value_counts()

x = counts.to_frame(name = 'count')

x.head(n=8)
fig, axs = plt.subplots()

fig.set_size_inches(10, 10, forward=True)

#x.head(n=8).values.transpose().tolist()[0]

#x.head(n=8).index.tolist()

axs.bar(x = x.head(n=8).index.tolist(), height = x.head(n=8).values.transpose().tolist()[0])