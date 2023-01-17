# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
season_stats = pd.read_csv('../input/Seasons_Stats.csv')
season_stats.sort_values(by='Year',ascending=False).head()
season_stats.loc[season_stats.Player== 'Ivica Zubac']
# Small Curious side thing, how has Lebron's Threes been progressing
lebron_stats = season_stats.loc[season_stats.Player == 'LeBron James']
lebron_stats
columnsarr = ['FG','FGA','FG%','3P%','3PA','2P%','2PA','FT%','AST','STL','BLK','TOV']
columnsarr[1]

plt.figure(figsize=[20,20])
for s in range(len(columnsarr)):
    plt.subplot(3,4,s+1)
    plt.scatter(lebron_stats.Year,lebron_stats[columnsarr[s]])
    plt.title(columnsarr[s])
sums_per_year = season_stats.groupby('Year').sum()

plt.figure(figsize=[20,20])
for s in range(len(columnsarr)):
    plt.subplot(3,4,s+1)
    plt.scatter(sums_per_year.index,sums_per_year[columnsarr[s]])
    plt.title(columnsarr[s])
season_stats = season_stats[np.isfinite(season_stats['3PA'])]

sums_per_year = season_stats.groupby('Year').sum()
mean_per_year = season_stats.groupby('Year').mean()
median_per_year = season_stats.groupby('Year').median()

plt.figure(figsize =[20,20])

#Total sum
plt.subplot(3,1,1)
threep = sums_per_year['3PA']
twop = sums_per_year['2PA']

plt.bar(sums_per_year.index,threep)
plt.bar(sums_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Sum of points')

#Mean
plt.subplot(3,1,2)
threep = mean_per_year['3PA']
twop = mean_per_year['2PA']

plt.bar(mean_per_year.index,threep)
plt.bar(mean_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Mean of points')

#Median
plt.subplot(3,1,3)
threep = median_per_year['3PA']
twop = median_per_year['2PA']

plt.bar(median_per_year.index,threep)
plt.bar(median_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Median of points')
# Lets check the points distribution for each position
posarray = ['PG','SG','PF','SF','C']

plt.figure(figsize =[20,20])

for pos in range(len(posarray)):
    pos_stats = season_stats.loc[season_stats.Pos == posarray[pos]]
    
    sums_per_year = pos_stats.groupby('Year').sum()
    mean_per_year = pos_stats.groupby('Year').mean()
    median_per_year = pos_stats.groupby('Year').median()
    
    plt.subplot(5,3,pos*3+1)
    plt.bar(sums_per_year.index,sums_per_year['3PA'])
    plt.bar(sums_per_year.index,sums_per_year['2PA'],bottom=sums_per_year['3PA'])
    plt.title('Sum' + " " + posarray[pos])
    plt.subplot(5,3,pos*3+2)
    plt.bar(sums_per_year.index,mean_per_year['3PA'])
    plt.bar(sums_per_year.index,mean_per_year['2PA'],bottom=mean_per_year['3PA'])
    plt.title('Mean' + " " + posarray[pos])
    plt.subplot(5,3,pos*3+3)
    plt.bar(sums_per_year.index,median_per_year['3PA'])
    plt.bar(sums_per_year.index,median_per_year['2PA'],bottom=median_per_year['3PA'])
    plt.title('Median' + " " + posarray[pos])

#Note to self: I should learn how to put bar plots side by side later to make the std bars visible

#It is hard to quantify how good a player is but we'll just use wins as a metric
center_stats = season_stats.loc[season_stats.Pos == 'C']
test_case = center_stats.sort_values('WS',ascending=False)[0:1000]
#Now lets see how these players shoot
test_case = test_case.groupby('Year').head(10).sort_values('Year')

sums_per_year = test_case.groupby('Year').sum()
mean_per_year = test_case.groupby('Year').mean()
median_per_year = test_case.groupby('Year').median()

plt.figure(figsize = (20,20))
plt.subplot(3,1,1)
plt.bar(sums_per_year.index,sums_per_year['3PA'])
plt.bar(sums_per_year.index,sums_per_year['2PA'],bottom=sums_per_year['3PA'])
plt.title('Sum' + " " + posarray[pos])

plt.subplot(3,1,2)
plt.bar(sums_per_year.index,mean_per_year['3PA'])
plt.bar(sums_per_year.index,mean_per_year['2PA'],bottom=mean_per_year['3PA'])
plt.title('Mean' + " " + posarray[pos])

plt.subplot(3,1,3)
plt.bar(sums_per_year.index,median_per_year['3PA'])
plt.bar(sums_per_year.index,median_per_year['2PA'],bottom=median_per_year['3PA'])
plt.title('Median' + " " + posarray[pos])

test = center_stats.loc[(center_stats.Year == 2015) | (center_stats.Year == 2016) | (center_stats.Year == 2017)]
plt.scatter(test.WS,test['3PA'])
plt.ylabel('3PA')
plt.xlabel('WS')
plt.title('WS vs 3PA')

plt.axvline(test.WS.median(),0,400)
testshot = test.loc[test.WS > test.WS.median()]
shooters = testshot.loc[testshot['3PA'] > 150]
nonshooters = testshot.loc[testshot['3PA'] <= 150]


print(str(len(shooters)/len(testshot)) + " " + "are shooters above WS mark") 
print(str(len(nonshooters)/len(testshot)) + " " + "are nonshooters above WS mark") 

testshot = test.loc[test.WS < test.WS.median()]
shooters = testshot.loc[testshot['3PA'] > 150]
nonshooters = testshot.loc[testshot['3PA'] <= 150]

print(str(len(shooters)/len(testshot)) + " " + "are shooters below WS mark") 
print(str(len(nonshooters)/len(testshot)) + " " + "are nonshooters below WS mark") 
player_data = pd.read_csv('../input/player_data.csv')
players = pd.read_csv('../input/Players.csv')
season_stats = pd.read_csv('../input/Seasons_Stats.csv')
player_data.head()
players.head()
player_data.position = player_data.position.apply(lambda s: str(s).split('-')[0])
player_data.head()
year_players = season_stats.loc[season_stats.Year==1950.0].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

for row in range(len(year_stats)):
    color = colors[player_data.loc[player_data.name == year_stats.loc[row].Player].position.values[0]]
    plt.scatter(year_stats.loc[row].height.item(),year_stats.loc[row].weight.item(),c=color)

#We have a problem, probably cause a null value in our data so lets take it out
player_data.position.isna().sum()
nullrow = player_data.loc[player_data.position.isnull()]
nullrow
player_data = pd.read_csv('../input/player_data.csv')
player_data = player_data.dropna(subset=['position'])
player_data.position = player_data.position.apply(lambda s: str(s).split('-')[0])
player_data.head()
#To make our legend after
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

year_players = season_stats.loc[season_stats.Year==1950.0].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

player_ready = player_data['name'].unique()

#Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
year_stats = year_stats.reset_index()
################################DEBUGGING CODE#############################################
#for row in range(len(year_stats)):
    #if year_stats.Player[row] in player_ready:
#    color = colors[player_data.loc[player_data.name == year_stats.loc[row].Player].position.values[0]]
#    plt.scatter(year_stats.loc[row].height.item(),year_stats.loc[row].weight.item(),c=color)
        
#custom_lines = [plt.scatter(0,0, color='Blue'),plt.scatter(0,0, color='Green'),plt.scatter(0,0, color='Red')]
#plt.legend(custom_lines,['C','G','P'])

#year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
#year_stats = year_stats.reset_index()
#year_stats
###############################################################################################
#Plotting
plt.scatter(year_stats.height,year_stats.weight,c=year_stats.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]))
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])
#Lets try to include season stats. Maybe WS could be a good metric. We average the WS of each player
metric = '2PA'
player_average = season_stats.groupby('Player')[metric].mean()
player_average_df = pd.DataFrame(player_average)
year_stats
year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
min(year_stats_WS[metric])
year_stats.WS = year_stats_WS[metric] + 5
#Plotting
plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.2*year_stats_WS[metric])
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])
#Now for the final lets include a scroll for the year
import matplotlib.animation as animation
from matplotlib.widgets import Slider

year = 1950
year_players = season_stats.loc[season_stats.Year==year].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

player_ready = player_data['name'].unique()

#Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
year_stats = year_stats.reset_index()
year_stats

#Size : Proportional to points**2
metric = 'PTS'
player_average = season_stats.groupby('Player')[metric].mean()
player_average_df = pd.DataFrame(player_average)
year_stats
year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
min(year_stats_WS[metric])
year_stats.WS = year_stats_WS[metric] + 5

#Plotting
plt.figure(figsize=(20,20))
plot = plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.001*(year_stats_WS[metric])**2)
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])

#Add a slider for the year

#season_stats.head()
i=0 #For the subplots
plt.figure(figsize=(20,20))

for year in range(1950,2016,4):

    #year = 1950
    year_players = season_stats.loc[season_stats.Year==year].Player.unique()
    year_stats = players.loc[players.Player.isin(year_players)]

    positions = player_data.position.unique()
    colors = dict(zip(positions,['Blue','Red','Green']))

    player_ready = player_data['name'].unique()

    #Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
    year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
    year_stats = year_stats.reset_index()
    year_stats

    #Size : Proportional to points**2
    metric = 'PTS'
    player_average = season_stats.groupby('Player')[metric].mean()
    player_average_df = pd.DataFrame(player_average)
    year_stats
    year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
    min(year_stats_WS[metric])
    year_stats.WS = year_stats_WS[metric] + 5

    #Plotting
    #plt.figure(figsize=(20,20))
    i = i + 1
    plt.subplot(4,5,i)
    plot = plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.0001*(year_stats_WS[metric])**2)
    plt.xlabel('height (cm)')
    plt.ylabel('weight (kg)')
    plt.ylim((70,130))
    plt.xlim((170,220))
    plt.title(year)

    red_patch = mpatches.Patch(color='red', label='Centers')
    blue_patch = mpatches.Patch(color='blue', label='Forwards')
    green_patch = mpatches.Patch(color='green', label='Guards')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
