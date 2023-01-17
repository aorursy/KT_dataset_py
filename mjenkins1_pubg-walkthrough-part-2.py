# Import necessary libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



warnings.filterwarnings("ignore")
# For plot sizes

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
os.listdir('../input')
# Load Part 1 data

data = pd.read_csv('../input/Training_Data_New.csv')

print("Done loading data from part 1")
# Let's review the features

data.columns
data['matchType'].value_counts().plot(kind='bar');
df_groups = (data.groupby('groupId', as_index=False).agg({'Id':'count', 'matchId':'count', 'assists':'sum', 'boosts':'sum',

                                'damageDealt':['sum', 'mean', 'max', 'min'], 'DBNOs':'sum', 'headshotKills':'sum',

                                'heals':['sum', 'mean'], 'killPlace':['mean', 'max', 'min'], 'killPoints':['mean', 'max', 'min'],

                                'kills':['sum', 'mean', 'max', 'min'],

                                'killStreaks':'mean', 'longestKill':'mean', 'matchDuration':['mean', 'min', 'max', 'sum'],

                                'maxPlace':['mean', 'min', 'max'], 'numGroups':['count','sum', 'mean', 'max', 'min'],

                                'revives':'sum', 'rideDistance':'max', 'roadKills':'sum', 'swimDistance':'max',

                                'teamKills':['sum', 'mean', 'max', 'min'], 'vehicleDestroys':'sum', 'walkDistance':['sum', 'mean', 'max', 'min'],

                                'weaponsAcquired':'sum','winPoints':['sum', 'mean', 'max', 'min'], 'winPlacePerc':'mean',

                                'killsPerMeter': 'mean', 'healsPerMeter': 'mean', 'killsPerHeal': 'mean',

                                'killsPerSecond': 'max', 'TotalHealsPerTotalDistance': 'max',

                                'killPlacePerMaxPlace': 'max'}).rename(columns={'Id':'teamSize'}).reset_index())
# Show changes

df_groups.head(5)
df_groups['teamSize'].describe()
df_groups['teamSize']['count'].value_counts()
df_groups[df_groups['teamSize']['count'] == 74]
df_groups.columns
sns.distplot(df_groups['kills']['mean']);
sns.distplot(df_groups['weaponsAcquired']['sum'], color='red');
sns.distplot(df_groups['damageDealt']['mean'], color='purple');
sns.distplot(df_groups['damageDealt']['sum'], color='purple');
sns.jointplot(df_groups['teamSize']['count'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkorange');
sns.jointplot(df_groups['winPoints']['mean'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='mediumseagreen');
sns.jointplot(df_groups['killPoints']['mean'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkblue');
sns.jointplot(df_groups['killPoints']['max'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkred');
# Save new grouped data

df_groups.to_csv(r'Training_Data_New_Groups.csv')