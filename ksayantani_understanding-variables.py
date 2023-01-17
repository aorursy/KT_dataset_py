# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import seaborn as sns

import random

import matplotlib.pyplot as plt

from matplotlib import patches

from kaggle.competitions import nflrush



# Any results you write to the current directory are saved as output.

%matplotlib inline

plt.rcParams['figure.figsize'] = (12, 6)
def calculate_distance(row):

    return np.sqrt((row['X'] - row['X_Rusher'])**2 + (row['Y'] - row['Y_Rusher'])**2)



def preprocess_play(df, play_id, cols=None):

    play_df = df.loc[df['PlayId'] == play_id].copy()

    play_df['Dis_from_Rusher'] = play_df.apply(lambda row: calculate_distance(row), axis=1)

    

    if cols:

        play_df = play_df[cols]

    return play_df



def create_field_points(points, deltas):

    s, e = points

    delta_x, delta_y = deltas

    x = []

    y = []

    for i in range(1, 10+1):

        x.extend([delta_x*i, delta_x*i])

        if i%2:

            y.extend([int(not (i%2))*delta_y, int(not (i%2))*delta_y + delta_y])

        else:

            y.extend([int(not (i%2))*delta_y, int(not (i%2))*delta_y - delta_y])

            

    return x, y
def draw_field(ax, play_id):

    rect = patches.Rectangle((0, 0), 120, 53.3, fc='seagreen', ec='w', zorder=0)

    ax.add_patch(rect)



    ez_left = patches.Rectangle((0, 0), 10, 53.3, fc='grey', zorder=1)

    ax.add_patch(ez_left)

    plt.text(5, 34, 'HOME ENDZONE', horizontalalignment='center', fontdict=dict(color='yellow', fontsize=14, rotation=90))



    ez_right = patches.Rectangle((110, 0), 10, 53.3, fc='grey', zorder=1)

    ax.add_patch(ez_right)

    plt.text(115, 35, 'VISITOR ENDZONE', horizontalalignment='center', fontdict=dict(color='yellow', fontsize=14, rotation=270))



    x_values, y_values = create_field_points((0, 0), (10, 53.3))

    plt.plot(x_values, y_values, color='white', linewidth=1)

    

    plt.xlim(0, 120)

    plt.ylim(0, 53.3)



    plt.text(10+0.5, 3, 'G', fontdict=dict(color='w', size=14))

    plt.text(10+0.5, 53.3-3, 'G', fontdict=dict(color='w', size=14, rotation=180))



    for i in range(1, 10):

        label = i*10 if i<=5 else ((10-i))*10

        plt.text((i+1)*10, 3, label, fontdict=dict(color='w', size=14))

        plt.text((i+1)*10, 53.3-3, label, fontdict=dict(color='w', size=14, rotation=180))



    plt.text(110-3, 3, 'G', fontdict=dict(color='w', size=14))

    plt.text(110-3, 53.3-3, 'G', fontdict=dict(color='w', size=14, rotation=180))



    plt.axis('off')

    plt.tight_layout()

    

    play_df = preprocess_play(df, play_id)

    plot_play(play_df)

    return fig, ax



def plot_play(play_df):

    play_df[play_df.Team == 'home'].plot('X', 'Y', kind='scatter', ax=ax, color='darkblue')

    play_df[play_df.Team == 'away'].plot('X', 'Y', kind='scatter', ax=ax, color='red')

    play_df[play_df.IsRusher].plot('X', 'Y', kind='scatter', ax=ax, s=50, color='yellow')



    yard_line = int(play_df['YardLine'].unique()[0])

    plt.plot([yard_line + 10]*2, [0, 53.3], '--', color='yellow', linewidth=2)

    plt.text(yard_line + 10 + 1, 10, 'YardLine at ' + str(yard_line), fontdict=dict(color='black', size=14))



def get_longest_running_plays(gameId, colors):

    game_df = df.loc[(df['GameId'] == gameId) & (df.IsRusher)].reset_index(drop=True)

    game_df['TimeSnap_next'] = list(game_df.loc[[min(len(game_df), i+1) for i, j in enumerate(game_df.index)]]['TimeSnap'].values)

    game_df['TimeSnap_next'] = game_df['TimeSnap_next'].dt.tz_localize('UTC')

    game_df['PlayDuration'] = (game_df['TimeSnap_next'] - game_df['TimeSnap']) / np.timedelta64(1,'s')

    

    game_df = game_df.loc[~game_df['PlayDuration'].isnull()].sort_values(by=['PlayDuration'], ascending=False)

    game_df['PlayDuration'] = game_df['PlayDuration'].astype(int)

    game_df['PlayDuration_scaled'] = (game_df['PlayDuration'] - game_df['PlayDuration'].min()) / (game_df['PlayDuration'].max() - game_df['PlayDuration'].min())



    red, green, blue = colors

    game_df['red'] = [red] * len(game_df)

    game_df['green'] = [green] * len(game_df)

    game_df['blue'] = [blue] * len(game_df)



    for c in ['red', 'green', 'blue']:

        game_df[c] = game_df.apply(lambda x: x[c] * (x['PlayDuration'] / game_df['PlayDuration'].max()), axis=1)

        game_df[c] = game_df[c].astype(int)



    game_df['color1'] = game_df.apply(lambda x: '#%02X%02X%02X' % (x.red, x.green, x.blue), axis=1)

    return game_df
target = 'Yards'

data_dir = '/kaggle/input/nfl-big-data-bowl-2020/'

r = lambda: random.randint(0,255)  #'#%02X%02X%02X' % (red, green, blue)
df = pd.read_csv(f"{data_dir}/train.csv", low_memory=False)

df['TimeHandoff'] = pd.to_datetime(df['TimeHandoff'])

df['TimeSnap'] = pd.to_datetime(df['TimeSnap'])
df['IsRusher'] = df['NflId'] == df['NflIdRusher']

rusher_df = df[df.IsRusher]

df = df.merge(rusher_df[['GameId', 'PlayId', 'X', 'Y']], how='left', on=['GameId', 'PlayId'], suffixes=('', '_Rusher'))
np.random.seed(0)

#game_id = np.random.choice(df['GameId'])

game_id = 2017121707



game_df = get_longest_running_plays(game_id, (34, 46, 213))



# d = game_df['PlayDuration'][:10]

# indx = np.arange(len(d))

# fig, ax = plt.subplots(figsize=(8, 6))

# ax.barh(indx, d, 0.85, color=list(game_df.loc[d.index, 'color1']))

# plt.yticks(indx, list(game_df.loc[d.index, 'PlayId']))

# ax.set_title(f'Play duration for game {game_id}')

# plt.show()
df.columns
fig, ax = plt.subplots(1, figsize=(10, 5))

fig, ax = draw_field(ax, 20171217070078)

plt.show()
play_df = preprocess_play(df, 20171217070078)

play_df[['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'Team', 

         'JerseyNumber', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr', 

         'PlayDirection']]
fig, ax = plt.subplots(figsize=(8, 5))

play_df[play_df.Team == 'home'].plot('X', 'Y', kind='scatter', ax=ax, s=50, color='blue')

play_df[play_df.Team == 'away'].plot('X', 'Y', kind='scatter', ax=ax, s=50, color='red')

plt.xlim(0, 100)

plt.ylim(0, 53.3)

plt.show()