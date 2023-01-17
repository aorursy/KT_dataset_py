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
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
df.head()
#Calculating headshot percentage 
df['headShotPercentage'] = df[['kills','headshotKills']].apply(lambda x: x['headshotKills']/x['kills'] if x['kills'] > 0 else 0, axis=1) #if kills are 0 returning 0 else 'heashotKills' by 'kills'
df[(df['headShotPercentage'] > 0.9) & (df['kills'] > 40)][['Id','boosts','heals','headshotKills','kills','winPlacePerc']] #finding outthe cheaters by searching the players with more than 40 kills and 90% of them are from headshots
#first Calculating the total Distance travelled by a player i.e, sum of 'rideDistance','swimDistance' and 'walkDistance'
df['totalDistance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
kills_without_movement = df[(df['totalDistance'] <= 0) & (df['kills'] > 10)]#Getting the players who had not moved but got kills more than 10
kills_without_movement['kills'].plot.hist(figsize=(12, 7)) #Plotting the graph to chech the frequency of kills
plt.xlabel('Kills')
plt.title('Frequency of kills by cheaters without movement')
weapons_without_movement = df[(df['totalDistance'] == 0) & (df['weaponsAcquired'] > 50)]#players who have more than 50 weapons and has not moved
fig, ax1 = plt.subplots(figsize=(12, 7))
sns.distplot(weapons_without_movement['weaponsAcquired'], ax=ax1)
plt.title('Frequency of weapons acquired by cheaters')
#Common player in both of previous observations
common_players = [player for player in kills_without_movement['Id'].unique() if player in weapons_without_movement['Id'].unique()]
common_players
matchType = kills_without_movement['matchType'].value_counts() #Getting the value count for each match in match type
matchType.drop(matchType[matchType == 0].index, axis=0, inplace=True)#dropping the matchType which has 0 value count.
matchType.plot.barh(figsize=(12,6))#plotting a bar plot
plt.xlabel('Numbers of players')
plt.ylabel('Match type')
plt.title('Match type cheaters are interested in')
fig, axes = plt.subplots(figsize=(20,5))
sns.boxplot(y='kills', x='matchType', data=df, ax= axes)#Plotting box plot between different match type and kills
plt.xticks(rotation=45)
plt.title('kills for different matches')
match_count = df.groupby(['matchType']).size().reset_index() #total size (count) of each match in match type
match_count[0] = match_count[0]/df.shape[0]*100  # playing % of each match
match_count.drop(match_count[match_count[0] < 1].index, axis=0, inplace=True) #Deleting the match which have playing percentage less than 1%
fig, ax1 = plt.subplots(figsize=(16,7))
ax1.pie(match_count[0], labels=match_count['matchType'], autopct='%1.2f%%', shadow=True, startangle=90)
match_to_keep = ['squad-fpp','squad','solo-fpp','solo','duo-fpp','duo']
most_match = df[df.matchType.isin(match_to_keep)] #Getting only matches with has playing % more than 1
fig, ax1 = plt.subplots(figsize=(16,7))
sns.pointplot(x='kills', y='winPlacePerc', hue='matchType', data=most_match, ax=ax1)
plt.title('Win percentage vs Kills for different matches')
sns.jointplot(y='boosts', x='winPlacePerc', data=df, color='#0066ff')#plotting for boosts
plt.title('Win Percentage vs Boosts')
fig, ax1 = plt.subplots(figsize=(13,5))# plotting for heals
sns.pointplot(x='heals', y='winPlacePerc', data=df, ax=ax1)
plt.xlim((0,30))
plt.title('win percentage vs heals')
fig, ax1 = plt.subplots(figsize=(14,6))
sns.distplot(df['totalDistance'], color='#1ab2ff', ax=ax1) #getting the distribution of total Distance
plt.title('Distribution of Total Distance')
fig, ax1 = plt.subplots(figsize=(14,6))
distance = df[df['totalDistance'] < 30000][['totalDistance','kills','winPlacePerc']] #Getting total distance less than 30 km (30000 m)
distance['totalDistance'] = distance['totalDistance'].apply(lambda x: np.around(x/1000)) #Converting the distance in Km and taking round of it
sns.lineplot(x='totalDistance', y='winPlacePerc', data=distance, color='#002233', ax=ax1)
plt.xlabel('Total Distance in Km')
plt.title('win percentage vs total distance')
bin_used = [0,10,20,30,40,55]
label_used = ['0-10','10-20','20-30','30-40','40-55']
categories = pd.cut(df['DBNOs'], bins=bin_used, labels=label_used, include_lowest=True) #Converting the knocks into bins

fig, ax1 = plt.subplots(figsize=(14,6))
sns.boxplot(y=df['winPlacePerc'], x=categories, ax=ax1)
plt.title('Win percentage vs Knocks')
plt.xlabel('Knocks')
df['killsPerDistance'] = df[['kills','totalDistance']].apply(lambda x : x['kills']/x['totalDistance'] if x['totalDistance'] > 0 else 0, axis=1) #If totalDistance is greater than 0 return 0 else return 'kills' by 'totalDistance'
df['totalBoosts/Heals'] = df['boosts'] + df['heals'] #sum of 'boosts' and 'heals'
df['damagePerKill'] = df.apply(lambda x: x['damageDealt']/x['kills'] if x['kills'] > 0 else 0, axis=1)#damage delt per kill
corr = df.corr() #Getting correlation
corr1 = np.abs(corr).nlargest(11, 'winPlacePerc') #absoluting the corr value so we can find the top 10 correlated columns
corr = corr.loc[corr1.index, corr1.index] #get highly correlated columns

fig, axes = plt.subplots(figsize=(13,6))
sns.heatmap(corr, annot=True, cmap='RdYlBu', ax=axes)
corr_col = df[corr.index] #Getting the correlated columns
sns.pairplot(corr_col)
