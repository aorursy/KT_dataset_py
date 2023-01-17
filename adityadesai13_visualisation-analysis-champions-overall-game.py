import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath = r'/kaggle/input/league-of-legends-ranked-matches/matches.csv'
matches = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/champs.csv'
champs = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/participants.csv'
participants = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/stats1.csv'
stats1 = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/stats2.csv'
stats2 = pd.read_csv(filepath)
stats = stats1.append(stats2)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/teamstats.csv'
teamstats = pd.read_csv(filepath)
df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))
df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))
df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))
df.head()
pd.options.display.max_columns = None
df.columns
# selecting a smaller subset of factors to look at
# using the describe function as a preliminary way to check the data seems alright (i.e. spot any errors)

df = df[['matchid', 'player', 'name', 'position', 'win', 'kills', 'deaths', 'assists', 'largestkillingspree', 'largestmultikill', 'longesttimespentliving', 'totdmgdealt', 'totdmgtochamp', 'totdmgtaken', 'turretkills', 'totminionskilled', 'goldearned', 'wardsplaced', 'duration', 'firstblood', 'seasonid']]
df.describe()
# roughly 150k rows removed i.e. ~ 15,000 matches

print(df.shape)
df = df[df['seasonid'] == 8]
df = df[df['duration'] >= 300]
df = df[df['totdmgdealt'] >= 0]

print(df.shape)
pd.options.display.float_format = '{:,.1f}'.format

df_win_rate = df.groupby('name').agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win matches'] /  df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate', ascending = False)
df_win_rate = df_win_rate[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]

# adding position that champions are most commonly played in

df_test = df.groupby('name').position.apply(lambda x: x.mode())
df_new = pd.merge(df_win_rate, df_test, how = 'left', on = ['name'], suffixes=('', '_y'))
df_new

print('Top 10 win rate')
print(df_new.head(10))
print('Bottom 10 win rate')
print(df_new.tail(10))
# this cell allows visulisation of the distribution of positions played for each champion. Shows most champions have a clear
# favorite lane that they are played in.

champ = 'Anivia'

df_test = df[df['name'] == champ]
print(df_test['position'].value_counts())
plt.figure(figsize=(12,8))
plt.title('Distribution of position played for '+str(champ))
plt.ylabel('# of games position picked')
sns.countplot(df_test['position'])
df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['matchid', 'player'], axis = 1)
corr = df_corr.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')
# average game time is ~ 31 minutes

df.loc[:,'duration'].describe()
df.hist(column='duration', bins=40)
df_corr_2 = df._get_numeric_data()
df_corr_2 = df_corr_2.drop(['matchid', 'player'], axis = 1)
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
corr = df_corr_2.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')
df_corr_3 = df._get_numeric_data()
df_corr_3 = df_corr_3.drop(['matchid', 'player'], axis = 1)
df_corr_3 = df_corr_3[df_corr_3['duration'] >= 2400]
corr = df_corr_3.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')