# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.style.use('ggplot')        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns',85)
pd.set_option('display.max_rows',85)
df = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')
#df.info()
#Variation of number of teams in top 10 leagues
df['League'].value_counts()
league_concat_str = (','.join(df['League'])).split(',')
league_dict_all = Counter(league_concat_str)
league_top10 = league_dict_all.most_common(10)
league_lst = [*map(lambda x: x[0],league_top10)]
season_list = df['Season'].unique()
fig,ax = plt.subplots(6,1)
fig.set_figheight(40)
fig.set_figwidth(15)
for league in league_lst:
    team_cnt = []
    player_cnt = []
    three_point_cnt = []
    point_cnt = []
    block_cnt = []
    foul_cnt = []
    for season in season_list:
        filt = ((df['League'] == league) & (df['Season'] == season))
        team_cnt.append(len(df.loc[filt,'Team'].unique()))
        player_cnt.append(len(df.loc[filt,'Player'].unique()))
        three_point_cnt.append(len(df.loc[filt,'3PM'].unique()))
        point_cnt.append(df.loc[filt,'PTS'].sum())
        block_cnt.append(df.loc[filt,'BLK'].sum())
        foul_cnt.append(df.loc[filt,'PF'].sum())
    ax[0].plot(season_list,team_cnt,label=league,linewidth=2,alpha=1,marker='.')
    ax[1].plot(season_list,player_cnt,label=league,linewidth=2,alpha=1,marker='.')
    ax[2].plot(season_list,three_point_cnt,label=league,linewidth=2,alpha=1,marker='.')
    ax[3].plot(season_list,point_cnt,label=league,linewidth=2,alpha=1,marker='.')
    ax[4].plot(season_list,block_cnt,label=league,linewidth=2,alpha=1,marker='.')
    ax[5].plot(season_list,foul_cnt,label=league,linewidth=2,alpha=1,marker='.')
ax[0].set_title('Number of teams in Top10 leagues over the years')
ax[0].set_xlabel('Seasons')
ax[0].set_ylabel('Counts')
ax[0].set_xticklabels(labels = season_list,rotation=90)
ax[0].legend()
ax[1].set_title('Number of players in Top10 leagues over the years')
ax[1].set_xlabel('Seasons')
ax[1].set_ylabel('Counts')
ax[1].set_xticklabels(labels = season_list,rotation=90)
ax[1].legend()
ax[2].set_title('Number of successful 3 pointers  in Top10 leagues over the years')
ax[2].set_xlabel('Seasons')
ax[2].set_ylabel('Counts')
ax[2].set_xticklabels(labels = season_list,rotation=90)
ax[2].legend()
ax[3].set_title('Nubmber of points scored in Top10 leagues over the years')
ax[3].set_xlabel('Seasons')
ax[3].set_ylabel('Counts')
ax[3].set_xticklabels(labels = season_list,rotation=90)
ax[3].legend()
ax[4].set_title('Number of blocks performed in Top10 leagues over the years')
ax[4].set_xlabel('Seasons')
ax[4].set_ylabel('Counts')
ax[4].set_xticklabels(labels = season_list,rotation=90)
ax[4].legend()
ax[5].set_title('Number of personal fouls occured in Top10 leagues over the years')
ax[5].set_xlabel('Seasons')
ax[5].set_ylabel('Counts')
ax[5].set_xticklabels(labels = season_list,rotation=90)
ax[5].legend()
plt.tight_layout()
plt.show()
#Variation of points/3pointers/fouls/block scored by a team over the years.
#Set the team name to see a perticular team stats
team = 'MIA'
three_point_cnt = []
point_cnt = []
block_cnt = []
foul_cnt = []
fig,ax = plt.subplots(4,1)
fig.set_figheight(20)
fig.set_figwidth(10)
for season in season_list:
    filt = ((df['Team'] == team) & (df['Season'] == season))
    three_point_cnt.append(df.loc[filt,'3PM'].sum())
    point_cnt.append(df.loc[filt,'FGM'].sum())
    block_cnt.append(df.loc[filt,'BLK'].sum())
    foul_cnt.append(df.loc[filt,'PF'].sum())
ax[0].plot(season_list,three_point_cnt)
ax[0].set_xlabel('Seasons')
ax[0].set_ylabel('Three pointers count')
ax[0].set_xticklabels(labels = season_list,rotation=90)
ax[0].set_title('Plot for three pointers scored by {}'.format(team))
ax[1].plot(season_list,point_cnt)
ax[1].set_xlabel('Seasons')
ax[1].set_ylabel('Points count')
ax[1].set_xticklabels(labels = season_list,rotation=90)
ax[1].set_title('Plot for total points scored by {}'.format(team))
ax[2].plot(season_list,block_cnt)
ax[2].set_xlabel('Seasons')
ax[2].set_ylabel('Blocks count')
ax[2].set_xticklabels(labels = season_list,rotation=90)
ax[2].set_title('Plot for blocks done by {}'.format(team))
ax[3].plot(season_list,foul_cnt)
ax[3].set_xlabel('Seasons')
ax[3].set_ylabel('Fouls count')
ax[3].set_xticklabels(labels = season_list,rotation=90)
ax[3].set_title('Plot for fouls by by {}'.format(team))
plt.tight_layout()
plt.show()
df.head()
team_list = ['OKC','MIA','LAL']
