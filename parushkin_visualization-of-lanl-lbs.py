import numpy as np

import pandas as pd

import json

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



PALETTE = ['#00d27a','#b88121', '#838280', '#8e5b3d', 'black']



df = pd.read_csv('../input/LANL agg LBs.csv', sep=';', index_col=0, parse_dates=['First step', 'Last step'])

df['activ'] = (df['Last step'] - df['First step']).astype('timedelta64[D]').astype(int)

df['strt activ'] = (df['First step'] - df['First step'].min()).astype('timedelta64[D]').astype(int)

df['end activ'] = (df['Last step'] - df['First step'].min()).astype('timedelta64[D]').astype(int)



df.head()
sns.palplot(sns.color_palette(PALETTE))
f, ax = plt.subplots(figsize=(16, 16))



g = sns.kdeplot(df['PVT rank'], df['PUB rank'], ax=ax, shade=True, n_levels=30)

sns.scatterplot(data=df, x='PVT rank', y='PUB rank', hue='Medal', palette=PALETTE, 

                  linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 10], ax=ax)



for i, j in enumerate([5, 19, 227, 454]):

    plt.plot([0, 4539], [j, j], color=PALETTE[i])
sns.set(rc={'figure.figsize':(15,15)})



tmp = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)

sns.scatterplot(data=df[:-7], x='PVT Score', y='PUB Score', hue='Medal', palette=PALETTE, 

                  linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 5])





plt.subplot2grid((3, 2), (2, 0))

sns.scatterplot(data=df[df['PVT rank'] < 455], x='PVT Score', y='PUB Score', hue='Medal', palette=PALETTE[:-1], 

                  linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50])

plt.title('**only medalists**')



plt.subplot2grid((3, 2), (2, 1))

sns.distplot(df[:-7]['PUB Score'], bins=150)

sns.distplot(df[:-7]['PVT Score'], bins=150)

plt.xlabel('PVT/PUB Score')

plt.legend(labels=['PUB', 'PVT'])



tmp.get_figure().subplots_adjust(hspace=0.5)
for i,j in enumerate(['PUB rank','PUB Score','PVT rank','PVT Score']):

    tmp = plt.subplot(2, 2, i+1)

    sns.scatterplot(data=df[:-15], x=j, y='Sh-up', hue='Medal', palette=PALETTE, 

                  linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 5])





tmp.get_figure().subplots_adjust(wspace=0.3)
f, ([ax_box1, ax_box2], [ax_hist1, ax_hist2]) = plt.subplots(2, 2, sharex=False, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(df["Sh-up"], ax=ax_box1)

sns.distplot(df["Sh-up"], bins=50, kde=False, ax=ax_hist1)

sns.boxplot(df["Sh-up"].abs(), ax=ax_box2)

sns.distplot(df["Sh-up"].abs(), bins=25, kde=False, ax=ax_hist2)



ax_box1.set(xlabel='')

ax_box2.set(xlabel='')

ax_hist2.set(xlabel='abs Sh-up')

f.subplots_adjust(hspace=0)

f.set_size_inches(15,5)
pd.crosstab(df['Team Size'], df['Medal'], margins=True)[['G+', 'G', 'S', 'B', '-', 'All']]
fig, ax = plt.subplots(3, 2)

y = [['Entries', 'activ'], ['Pub score step', 'PUB rank'], ['strt activ', 'end activ']]

for j in range(len(ax)):

    for i in range(len(ax[j])):

        sns.catplot(y=y[j][i], x='Team Size', data=df, kind="boxen", ax=ax[j][i])

        plt.close(2)
fig, ax = plt.subplots(3, 2)

y = [['Entries', 'activ'], ['Pub score step', 'PUB rank'], ['strt activ', 'end activ']]

for j in range(len(ax)):

    for i in range(len(ax[j])):

        sns.catplot(y=y[j][i], x='Medal', data=df, kind="boxen", palette=PALETTE, ax=ax[j][i])

        plt.close(2)

plt.vlines(x=df['PVT rank'].iloc[:454], ymin=df['strt activ'].iloc[:454], ymax=df['end activ'].iloc[:454], color='b', alpha=0.2)



ax = sns.scatterplot(data=df.iloc[:454], x='PVT rank', y='end activ', hue='Medal', palette=PALETTE[:-1], 

                  linewidth=0, alpha=1, size='Medal', sizes=[150, 100, 50, 50])





ax.set(ylabel='number of days since the beginning of the competition', title='Medalists activity')

plt.show()
for i, x in enumerate(['PVT Score', 'PUB Score']):

    for j, y in enumerate(['Pub score step', 'Entries', 'activ']):

        plt.subplot(3, 2, 2*j+i+1)

        sns.scatterplot(data=df[:-7], x=x, y=y, hue='Medal', palette=PALETTE, 

                          linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 5])
for i, x in enumerate(['PVT Score', 'PUB Score']):

    for j, y in enumerate(['Pub score step', 'Entries', 'activ']):

        plt.subplot(3, 2, 2*j+i+1)

        ax = sns.scatterplot(data=df[((df['PVT rank'] < 455) | (df['PUB rank'] < 455))], x=x, y=y, hue='Medal', 

                        palette=PALETTE, 

                          linewidth=0, alpha=0.7, size='Medal', sizes=[200, 100, 50, 25, 5])
for i, x in enumerate(['Pub score step', 'Entries', 'activ', 'Team Size', 'strt activ', 'end activ']):

    plt.subplot(3, 2, i+1)

    sns.scatterplot(data=df, x=x, y='Sh-up', hue='Medal', palette=PALETTE, 

                      linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 5])
for i, x in enumerate(['Pub score step', 'Entries', 'activ', 'Team Size', 'strt activ', 'end activ']):

    plt.subplot(3, 2, i+1)

    sns.scatterplot(data=df[((df['PVT rank'] < 455) | (df['PUB rank'] < 455))], x=x, y='Sh-up', hue='Medal', palette=PALETTE, 

                      linewidth=0, alpha=0.7, size='Medal', sizes=[150, 100, 50, 50, 5])