import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("husl")
sns.set_style('white')
train_df = pd.read_csv("../input/train_V2.csv")
print("Size of train dataset : " + str(len(train_df)))
train_df.head()
features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills', 'killStreaks', 
            'longestKill', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired']
infos = ['matchDuration', 'matchType', 'maxPlace', 'numGroups']
ELO = ['rankPoints', 'killPoints', 'winPoints']
label = ['winPlacePerc']
sample = train_df.sample(100000)

f,ax = plt.subplots(figsize=(15, 12))
sns.heatmap(sample[ELO + features + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
plt.figure(figsize=(15,10))
train_df['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
game_index_dic = {"solo": 1, "solo-fpp": 1, 'normal-solo': 1, "normal-solo-fpp": 1,
       "duo": 2, "duo-fpp": 2, 'normal-duo': 2,"normal-duo-fpp": 2,
       "squad": 3, "squad-fpp": 3, 'normal-squad': 3,"normal-squad-fpp": 3,
       "crashfpp" :4, "crashtpp" :4,
       "flarefpp" :5, "flaretpp":5
      }

game_name_dic = {"solo": "solo", "solo-fpp": "solo", 'normal-solo': "solo", "normal-solo-fpp": "solo",
                   "duo": "duo", "duo-fpp": "duo", 'normal-duo': "duo","normal-duo-fpp": "duo",
                   "squad": "squad", "squad-fpp": "squad", 'normal-squad': "squad","normal-squad-fpp": "squad",
                   "crashfpp": "crash", "crashtpp": "crash",
                   "flarefpp": "flare", "flaretpp": "flare"
      }
train_df['matchTypeName'] = train_df['matchType'].apply(lambda x: game_name_dic[x])
train_df['matchType'] = train_df['matchType'].apply(lambda x: game_index_dic[x])
f, ax = plt.subplots(figsize=(12, 8))
ax.set(yscale="log")
train_df['matchTypeName'].value_counts().sort_values(ascending=False).plot.bar(ax=ax)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(train_df['matchDuration'])
plt.show()
def get_map(x):
    if x > 1600:
        return 0
    else: 
        return 1
train_df['miniRoyale'] = train_df['matchDuration'].transform(get_map)
data = train_df
f, ax = plt.subplots(figsize=(12, 8))
ax.set(yscale="log")
sns.countplot(x='miniRoyale', hue='matchTypeName', ax=ax, data=data, palette=sns.color_palette(n_colors=5))
plt.show()
agg = train_df.groupby(['matchId', 'groupId']).size().to_frame('teamSize')
train_df = train_df.merge(agg, how='left', on=['matchId', 'groupId'])
train_df.head()
data = train_df[['teamSize']].copy()
data.loc[data['teamSize'] >= 9] = '9+'
plt.figure(figsize=(15,10))
sns.countplot(data['teamSize'].astype('str').sort_values())
plt.show()
size_dic = {1:1, 2:2, 3:4, 4:4, 5:4}
sample = train_df[train_df['matchType'] == 1]
plt.figure(figsize=(15, 6))
sns.countplot(sample['teamSize'])
plt.show()
sample = train_df[train_df['matchId'] == "6dc8ff871e21e6"]
plt.figure(figsize=(10,4))
sns.countplot(sample['teamSize'])
plt.show()
fig,ax = plt.subplots(figsize=(12,8))
ax.set_xticklabels([])
sns.scatterplot(x='Id', y='winPlacePerc', hue='teamSize', data=sample, ax=ax)
plt.show()
train_df['matchTeamSize'] = train_df['teamSize'] / train_df['matchType'].apply(lambda x: size_dic[x])
train_df['matchTeamSize'] = train_df['matchTeamSize'].apply(lambda x: max(2, x))
train_df['skill'] = train_df['headshotKills'] + 0.01 * train_df['longestKill'] - train_df['teamKills']/(train_df['kills']+1)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='skill', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
train_df['hsRatio'] = train_df['headshotKills'] / train_df['kills']
train_df['hsRatio'].fillna(0, inplace=True)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='hsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
def transform_hsRatio(x):
    if x == 1 or x == 0:
        return 0.5
    else: 
        return x
train_df['hsRatio'] = train_df['hsRatio'].apply(transform_hsRatio)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='hsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette("husl", n_colors=5))
plt.show()
train_df['distance'] = (train_df['walkDistance'] + 0.4 * train_df['rideDistance'] + train_df['swimDistance'])/train_df['matchDuration']
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='distance', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
train_df['boostsRatio'] = train_df['boosts']**2 / train_df['walkDistance']**0.5
train_df['boostsRatio'].fillna(0, inplace=True)
train_df['boostsRatio'].replace(np.inf, 0, inplace=True)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='boostsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
train_df['healsRatio'] = train_df['heals'] / train_df['matchDuration']**0.1
train_df['healsRatio'].fillna(0, inplace=True)
train_df['healsRatio'].replace(np.inf, 0, inplace=True)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='healsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
train_df['killsRatio'] = train_df['kills'] / train_df['matchDuration']**0.1
train_df['killsRatio'].fillna(0, inplace=True)
train_df['killsRatio'].replace(np.inf, 0, inplace=True)
data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='killsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()
engineered = ['matchTeamSize', 'skill', 'hsRatio', 'distance', 'boostsRatio', 'healsRatio', 'killsRatio']
sample = train_df.sample(100000)

f,ax = plt.subplots(figsize=(15, 12))
sns.heatmap(sample[engineered + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()