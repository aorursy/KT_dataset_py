# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load csv file to dataframe

champs_df = pd.read_csv("../input/nba-finals-team-stats/championsdata.csv")

runnerups_df = pd.read_csv("../input/nba-finals-team-stats/runnerupsdata.csv")
champs_df.head()
# combine champs df and runnerups df

df = pd.concat([champs_df, runnerups_df])

df.sort_values(by=['Year', 'Game'], inplace=True)

df = df.fillna(0)
df.head()
# calculate correlation matrix



# plt.figure(figsize=(10,10))

# corr = df.corr()

# ax = sns.heatmap(

#     corr, 

#     vmin=-1, vmax=1, center=0,

#     cmap=sns.diverging_palette(20, 220, n=200),

#     square=True

# )

# ax.set_xticklabels(

#     ax.get_xticklabels(),

#     rotation=45,

#     horizontalalignment='right'

# );
# Offensive Stats

off_stats = ['Year', 'Win', 'FG', 'FGA', 'FGP', 'TP', 'TPA', 'TPP', 'FT', 'FTA', 'FTP', 'ORB', 'AST', 'PTS']

df_off = df[off_stats]



df_off_win = df_off.query('Win == 1')

df_off_lose = df_off.query('Win == 0')

df_off_lose 
# Offensive Stats Correlation Matrix

plt.figure(figsize=(10,10))

corr = df_off.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
corr
# Average PTS in Finals games per year

means_wins = df_off_win.groupby('Year')['PTS'].mean()

means_loss = df_off_lose.groupby('Year')['PTS'].mean()

plt.xlabel('Years')

plt.ylabel('PTS')

plt.title('Average PTS in the NBA Finals from 1980-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average FG% in Finals games per year

means_wins = df_off_win.groupby('Year')['FGP'].mean()

means_loss = df_off_lose.groupby('Year')['FGP'].mean()

plt.xlabel('Years')

plt.ylabel('FG%')

plt.title('Average FG% in the NBA Finals from 1980-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average AST in Finals games per year

means_wins = df_off_win.groupby('Year')['AST'].mean()

means_loss = df_off_lose.groupby('Year')['AST'].mean()

plt.xlabel('Years')

plt.ylabel('AST')

plt.title('Average AST in the NBA Finals from 1980-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average AST in Finals games per year

df_win_2010s = df_off_win.query('Year <= 1990')

df_loss_2010s = df_off_lose.query('Year <= 1990')



means_wins = df_win_2010s.groupby('Year')['AST'].mean()

means_loss = df_loss_2010s.groupby('Year')['AST'].mean()

plt.xlabel('Years')

plt.ylabel('AST')

plt.title('Average AST in the NBA Finals')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average AST in Finals games per year

df_win_2010s = df_off_win.query('Year >= 1990 & Year <= 1995')

df_loss_2010s = df_off_lose.query('Year >= 1990 & Year <= 1995')



means_wins = df_win_2010s.groupby('Year')['AST'].mean()

means_loss = df_loss_2010s.groupby('Year')['AST'].mean()

plt.xlabel('Years')

plt.ylabel('AST')

plt.title('Average AST in the NBA Finals')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average AST in Finals games per year

df_win_2010s = df_off_win.query('Year >= 1995')

df_loss_2010s = df_off_lose.query('Year >= 1995')



means_wins = df_win_2010s.groupby('Year')['AST'].mean()

means_loss = df_loss_2010s.groupby('Year')['AST'].mean()

plt.xlabel('Years')

plt.ylabel('AST')

plt.title('Average AST in the NBA Finals')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='best')
# Average 3PT Attempts in Finals games per year

means_wins = df_off_win.groupby('Year')['TPA'].mean()

means_loss = df_off_lose.groupby('Year')['TPA'].mean()

plt.xlabel('Years')

plt.ylabel('3PTA')

plt.title('Average 3PT Attempts in the NBA Finals from 1980-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='upper center')
# Average 3PT Made in Finals games per year

means_wins = df_off_win.groupby('Year')['TP'].mean()

means_loss = df_off_lose.groupby('Year')['TP'].mean()

plt.xlabel('Years')

plt.ylabel('3PT')

plt.title('Average 3PT Made in the NBA Finals from 1980-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='lower right')
# Average 3PT Made in Finals games per year

df_win_2010s = df_off_win.query('Year >= 2000 & Year <= 2010')

df_loss_2010s = df_off_lose.query('Year >= 2000 & Year <= 2010')



means_wins = df_win_2010s.groupby('Year')['TP'].mean()

means_loss = df_loss_2010s.groupby('Year')['TP'].mean()

plt.xlabel('Years')

plt.ylabel('3PTA')

plt.title('Average 3PT Made in the NBA Finals from 2000-2010')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='lower right')
# Average 3PT Made in Finals games per year

df_win_2010s = df_off_win.query('Year >= 2010')

df_loss_2010s = df_off_lose.query('Year >= 2010')



means_wins = df_win_2010s.groupby('Year')['TP'].mean()

means_loss = df_loss_2010s.groupby('Year')['TP'].mean()

plt.xlabel('Years')

plt.ylabel('3PTA')

plt.title('Average 3PT Made in the NBA Finals from 2010-2018')

plt.plot(means_wins, label="Win")

plt.plot(means_loss, label="Loss")

plt.legend(loc='lower right')