import numpy as np

import pandas as pd

import gc, os, sys

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

%matplotlib inline
def dtypes():

    train = pd.read_csv("../input/train_2.csv", index_col='Page', nrows=2)

    return {c:np.float32 for c in train.columns}
train = np.log1p(pd.read_csv("../input/train_2.csv", index_col='Page', dtype=dtypes()))

train.columns = train.columns.astype('datetime64[ns]')

train.fillna(0, inplace=True)

train.sort_index(inplace=True)

train.head()
def save_plot(fname, df):

    dat = df.values

    dat = (dat / np.max(dat)) * 255.

    print (fname, np.min(dat), np.max(dat))

    imageio.imsave(fname, dat.astype(np.uint8))



#save_plot('testplot.png', train.loc[stats.sort_values('sum').tail(900).index])
def substr(s):

    return train.loc[train.index.str.contains(s)]



def substrs(l):

    return pd.concat((substr(s) for s in l))
# Change the palette for whole notebook here...

cmap = 'magma'



def show_plot(df):

    dat = df.values

    dat = (dat / np.max(dat)) * 255.



    fig, ax = plt.subplots()

    fig.set_size_inches(18, 8)

    ax.xaxis_date()

    ax.yaxis.tick_left()

    ax.grid('off')

    cols = df.columns

    x_lims = [ cols.min(), cols.max() ]

    x_lims = mdates.date2num(x_lims)

    y_lims = [0, df.shape[0]]

    plt.imshow(dat, cmap=cmap, aspect='equal', extent=[x_lims[0], x_lims[1],  y_lims[0], y_lims[1]])

    date_format = mdates.DateFormatter('%y-%m-%d')

    ax.xaxis.set_major_formatter(date_format)

    plt.show()



def show_plot_for_index(idx):

    print(idx)

    show_plot(train.loc[idx])
show_plot(train.loc[train.index.str.startswith('API:')])
train.max(1).describe()
show_plot(train.loc[train.index.str.contains('Olympic')])
show_plot(train.loc[train.index.str.contains('Super_Bowl')])
show_plot(train.loc[train.index.str.contains('Special:WhatLinksHere')])
show_plot(substr('/featured/201'))
show_plot(substrs(('Game_of_Thrones', 'Walking_Dead')))
show_plot(substr('serie_de_televis'))
show_plot(substr('Topic:'))
show_plot(substrs(('Lewis_Hamilton', 'Nico_Rosberg', 'Max_Verstappen', 'Niki_Lauda')))
show_plot(train.loc[train.index.str.startswith('Category:Deletion')])
show_plot(train.loc[train.index.str.startswith('Halloween')])
show_plot(train.loc[train.index.str.startswith('Fußball-')])
show_plot(substr('File:Flag_'))
show_plot(substr('Help:Categories'))
show_plot(substr('Leicester'))
show_plot(substr('United_Kingdom'))
show_plot(substr('User:GoogleAnalitycsRoman'))
show_plot(substr('Prince'))
show_plot(substrs(['Star_Wars', 'スター']))
def do_sample(df, n):

    return df.sample(n, random_state=42)
show_plot(do_sample(substr('_es.wiki'), 200))
show_plot(do_sample(substr('_ru.wiki'), 200))
show_plot(do_sample(substr('_fr.wiki'), 200))
show_plot(do_sample(substr('_de.wiki'), 200))
show_plot(do_sample(substr('_ja.wiki'), 200))
show_plot(do_sample(substr('_zh.wiki'), 200))
stats = train.sum(1).to_frame('sum')

stats['mean'] = train.mean(1)

stats['max'] = train.max(1)

stats['min'] = train.min(1)

stats.describe()
date_of_interest = '2016-08-22'

ser = (train.loc[:, date_of_interest] / stats['sum']).sort_values().dropna()

show_plot(train.loc[ser[-300:].index])
ser = (stats['max'] / stats['mean']).sort_values().dropna()

show_plot(train.loc[ser[-300:].index])
ser = (stats['max'] / stats['mean']).sort_values().dropna()

show_plot_for_index (ser[:300].index)
ser = stats['min'].sort_values().dropna()

show_plot_for_index(ser[-200:].index)
import seaborn as sns
def show_heatmap(df):

    g = sns.clustermap(df.corr(), figsize=(12,12))

    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    plt.show()
show_heatmap(substrs(('Lewis_Hamilton', 'Nico_Rosberg', 'Max_Verstappen', 'Niki_Lauda')).T)
show_heatmap(substr('File:Flag').sample(25, random_state=4242).T)