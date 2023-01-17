# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df= pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")

test= pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

train= pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")
train.head()
train.shape
train.info()
train.describe()
missing_data=train.isna().sum()

missing_data.columns=['Missing data']
m_types = train.loc[:,"matchType"].value_counts().to_frame().reset_index()

m_types.columns = ["Type","Count"]

m_types
number_of_matches = train.loc[:,"matchId"].nunique()

number_of_matches
plt.figure(figsize=(15,8))

ticks = m_types.Type.values

ax = sns.barplot(x="Type", y="Count", data=m_types)

ax.set_xticklabels(ticks, rotation=60, fontsize=14) # this helps us in rotating the font.

ax.set_title("Match types")

plt.show()
m_types2 = train.loc[:,"matchType"].value_counts().to_frame()

aggregated_squads = m_types2.loc[["squad-fpp","squad","normal-squad-fpp","normal-squad"],"matchType"].sum()

aggregated_duos = m_types2.loc[["duo-fpp","duo","normal-duo-fpp","normal-duo"],"matchType"].sum()

aggregated_solo = m_types2.loc[["solo-fpp","solo","normal-solo-fpp","normal-solo"],"matchType"].sum()

aggregated_mt = pd.DataFrame([aggregated_squads,aggregated_duos,aggregated_solo], index=["squad","duo","solo"], columns =["count"])

aggregated_mt
fig1, ax1 = plt.subplots(figsize=(5, 5))

labels = ['squad', 'duo', 'solo']

wedges, texts, autotexts = ax1.pie(aggregated_mt["count"],textprops=dict(color="w"), autopct='%1.1f%%')

ax1.legend(wedges, labels,title="Types",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight="bold")

plt.show()
plt.figure(figsize=(15,8))

Aa=sns.distplot(train['numGroups'])

plt.title('Number of groups')

plt.show()
plt.figure(figsize=(15,8))

Az=sns.boxplot(x='kills',y='damageDealt',data=train)

plt.title('Damage Dealth VS Number of kills')

plt.show()
train[train['kills']>60][["Id","assists","damageDealt","headshotKills","kills","longestKill"]]
headshots = train[train['headshotKills']>0]

plt.figure(figsize=(15,5))

sns.countplot(headshots['headshotKills'].sort_values())

print("Maximum number of headshots that the player scored: " + str(train["headshotKills"].max()))
headshots = train[train['DBNOs']>0]

plt.figure(figsize=(15,5))

sns.countplot(headshots['DBNOs'].sort_values())

print("Mean number of DBNOs that the player scored: " + str(train["DBNOs"].mean()))
plt.figure(figsize=(15,8))

ax2 = sns.boxplot(x="DBNOs",y="kills", data = train)

ax2.set_title("Number of DBNOs vs. Number of Kills")

plt.show()
dist = train[train['longestKill']<200]

#plt.rcParams['axes.axisbelow'] = True

dist.hist('longestKill', bins=20, figsize = (16,8))

plt.show()
print("Average longest kill distance a player achieve is {:.1f}m, 95% of them not more than {:.1f}m and a maximum distance is {:.1f}m." .format(train['longestKill'].mean(),train['longestKill'].quantile(0.95),train['longestKill'].max()))
walk0 = train["walkDistance"] == 0

print('number of players dint walk at all:',walk0.sum())
ride0 = train["rideDistance"] == 0

print('number of players who dint ride at all:',ride0.sum())
swim0=train['swimDistance']==0

print('number of players who dint swim at all',swim0.sum())
travel_dist = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]

travel_dist = travel_dist[travel_dist<5000]

travel_dist.hist(bins=40, figsize = (15,10))

plt.show()
top10 = train[train["winPlacePerc"]>0.9]

print("TOP 10% overview\n")

print("Average number of kills: {:.1f}\nMinimum: {}\nThe best: {}\n95% of players within: {} kills." 

      .format(top10["kills"].mean(), top10["kills"].min(), top10["kills"].max(),top10["kills"].quantile(0.95)))
print("On average the best 10% of players have the longest kill at {:.3f} meters, and the best score is {:.1f} meters."

      .format(top10["longestKill"].mean(), top10["longestKill"].max()))