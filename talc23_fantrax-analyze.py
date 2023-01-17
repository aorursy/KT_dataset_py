# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/nbadates"))

# Any results you write to the current directory are saved as output.
players = pd.read_csv('../input/fantrax2/Fantrax-players.csv')
dates = pd.read_csv('../input/nbadates/nba-2018-UTC-08.csv')
dates.drop(['Round Number', 'Location', 'Result'],axis=1, inplace=True)
dates['Date'] = dates['Date'].astype('datetime64')
periodGames = dates[(dates['Date'] >= pd.Timestamp(year=2018, month=10, day=16)) & (dates['Date'] <= pd.Timestamp(year=2018, month=10, day=22))]
periodGames
nameToAbv = {}
nameToAbv['Hawks'] = 'ATL'
nameToAbv['Nets'] = 'BKN'
nameToAbv['Celtics'] = 'BOS'
nameToAbv['Hornets'] = 'CHA'
nameToAbv['Bulls'] = 'CHI'
nameToAbv['Cavaliers'] = 'CLE'
nameToAbv['Mavericks'] = 'DAL'
nameToAbv['Nuggets'] = 'DEN'
nameToAbv['Pistons'] = 'DET'
nameToAbv['Warriors'] = 'GS'
nameToAbv['Rockets'] = 'HOU'
nameToAbv['Pacers'] = 'IND'
nameToAbv['Clippers'] = 'LAC'
nameToAbv['Lakers'] = 'LAL'
nameToAbv['Grizzlies'] = 'MEM'
nameToAbv['Heat'] = 'MIA'
nameToAbv['Bucks'] = 'MIL'
nameToAbv['Timberwolves'] = 'MIN'
nameToAbv['Pelicans'] = 'NO'
nameToAbv['Knicks'] = 'NY'
nameToAbv['Thunder'] = 'OKC'
nameToAbv['Magic'] = 'ORL'
nameToAbv['76ers'] = 'PHI'
nameToAbv['Suns'] = 'PHO'
nameToAbv['Blazers'] = 'POR'
nameToAbv['Kings'] = 'SAC'
nameToAbv['Spurs'] = 'SA'
nameToAbv['Raptors'] = 'TOR'
nameToAbv['Jazz'] = 'UTA'
nameToAbv['Wizards'] = 'WAS'



periodGames['Home Team'] = periodGames['Home Team'].apply(lambda x:nameToAbv[x.split()[-1:][0]])
periodGames['Away Team'] = periodGames['Away Team'].apply(lambda x:nameToAbv[x.split()[-1:][0]])
periodGames
playersGrouped = players[players['Status']!='FA'].groupby(by='Status',as_index=False).mean()
playersGrouped.drop(['Rk','GP'], axis=1,inplace=True)
labels = playersGrouped.columns[1:]
playersGroupedNorm = playersGrouped.copy()
for label in labels:
    if label.endswith('%'):
        continue
    playersGroupedNorm[label]-=playersGroupedNorm[label].min()
    playersGroupedNorm[label]/=playersGroupedNorm[label].max()
    if label == 'TO':
        playersGroupedNorm[label]=1-playersGroupedNorm[label]
playersGroupedNorm.describe()
periodGames.groupby('Home Team',as_index=False).count().drop('Away Team', axis=1)
gamesPerTeam = periodGames.groupby('Home Team',as_index=False).count().drop('Away Team', axis=1)
gamesPerTeam = gamesPerTeam.merge(periodGames.groupby('Away Team',as_index=False).count().drop('Home Team', axis=1),how='outer', left_on='Home Team', right_on='Away Team')
gamesPerTeam['Team'] = gamesPerTeam['Home Team']
gamesPerTeam['Team'].fillna(gamesPerTeam['Away Team'], inplace=True)
gamesPerTeam['Date_x'].fillna(0, inplace=True)
gamesPerTeam['Date_y'].fillna(0, inplace=True)
gamesPerTeam.drop(['Home Team', 'Away Team'],inplace=True,axis=1)
gamesPerTeam['Total'] = gamesPerTeam['Date_x']+gamesPerTeam['Date_y']
gamesPerTeam.drop(['Date_x', 'Date_y'],inplace=True,axis=1)
gamesPerTeam
playersStatsFactorizedByGames = players[players['Status']!='FA'].copy()
playersStatsFactorizedByGames = playersStatsFactorizedByGames.merge(gamesPerTeam, how='outer', on='Team')
playersStatsFactorizedByGames.drop(['Rk','GP','ADP','%D','Score','Opponent'], axis=1,inplace=True)

colsToCalc = ['FG%', '3PTM', 'FT%', 'PTS',
       'REB', 'AST', 'ST', 'BLK', 'TO']
for col in colsToCalc:
    playersStatsFactorizedByGames[col + '_Fact']=playersStatsFactorizedByGames[col]*playersStatsFactorizedByGames['Total']
playersStatsFactorizedByGames.drop(colsToCalc, axis=1, inplace=True)
playersStatsFactorizedByGamesGrouped = playersStatsFactorizedByGames.groupby('Status', as_index=False).sum()
playersStatsFactorizedByGamesGrouped['FT%_Fact']/=playersStatsFactorizedByGamesGrouped['Total']
playersStatsFactorizedByGamesGrouped['FG%_Fact']/=playersStatsFactorizedByGamesGrouped['Total']
playersStatsFactorizedByGamesGrouped.drop('Total', axis=1, inplace=True)
playersStatsFactorizedByGamesGrouped.columns

scale=True
if scale:
    labels= [ '3PTM_Fact','PTS_Fact', 'REB_Fact',
       'AST_Fact', 'ST_Fact', 'BLK_Fact', 'TO_Fact']
    for label in labels:
        playersStatsFactorizedByGamesGrouped[label]-=playersStatsFactorizedByGamesGrouped[label].min()
        playersStatsFactorizedByGamesGrouped[label]/=playersStatsFactorizedByGamesGrouped[label].max()
        if label == 'TO':
            playersStatsFactorizedByGamesGrouped[label]=1-playersStatsFactorizedByGamesGrouped[label]
playersStatsFactorizedByGamesGrouped.describe()
playersStatsFactorizedByGamesGrouped.head()
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
def f(x):
    return x

import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import Checkbox, interactive
from IPython.display import display

l = playersStatsFactorizedByGamesGrouped['Status']
chk = [Checkbox(description=a) for a in l]

def updatePlot(**k):
    #print(k)
    teamsToShow = []
    for key in k.keys():
        if k[key] is True:
            teamsToShow.append(key)
    if len(teamsToShow) == 0:
        return
    print(teamsToShow)
    fig=plt.figure(figsize=(12, 10))
    labels=np.array(playersStatsFactorizedByGamesGrouped.columns[1:])
    ax = fig.add_subplot(111, polar=True)
    ax.set_title("Compare")
    i=0
    for team in teamsToShow:
        color='C'+str(i)
        stats=playersStatsFactorizedByGamesGrouped[playersStatsFactorizedByGamesGrouped['Status']==team][labels].values[0]
        #print(stats)
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))

        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, stats, 'o-', linewidth=2, label=team)
        ax.fill(angles, stats, alpha=0.25, color=color)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        #ax.set_title([name], color=color)
        ax.grid(True)
        i+=1
    legend = ax.legend(loc=1)

interact(updatePlot, **{c.description: c.value for c in chk})

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
def f(x):
    return x

import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import Checkbox, interactive
from IPython.display import display

l = playersGrouped['Status']
chk = [Checkbox(description=a) for a in l]

def updatePlot(**k):
    #print(k)
    teamsToShow = []
    for key in k.keys():
        if k[key] is True:
            teamsToShow.append(key)
    if len(teamsToShow) == 0:
        return
    print(teamsToShow)
    fig=plt.figure(figsize=(12, 10))
    labels=np.array(playersGrouped.columns[1:])
    ax = fig.add_subplot(111, polar=True)
    ax.set_title("Compare")
    i=0
    for team in teamsToShow:
        color='C'+str(i)
        stats=playersGroupedNorm[playersGroupedNorm['Status']==team][labels].values[0]
        #print(stats)
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))

        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, stats, 'o-', linewidth=2, label=team)
        ax.fill(angles, stats, alpha=0.25, color=color)
        ax.set_thetagrids(angles * 180/np.pi, labels)
        #ax.set_title([name], color=color)
        ax.grid(True)
        i+=1
    legend = ax.legend(loc=1)

interact(updatePlot, **{c.description: c.value for c in chk})



