# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
%pylab inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
import warnings
warnings.filterwarnings(action='ignore')
# Any results you write to the current directory are saved as output.
plt.figure(figsize = (20, 20))
df = pd.read_csv('../input/train_V2.csv')
df.columns
df.info()
def plot_counts(p):
    plt.figure(figsize = (20, 15))
    sns.countplot(df[p]).set_title(p)
    plt.show()
def plot_dists(p, b = 50, kde_flag = True, rug_flag = False):
    plt.figure(figsize = (20, 15))
    sns.distplot(df[p], kde= kde_flag, rug = rug_flag, bins = b).set_title(p)
    plt.show()
def plot_scatters(x, y, title):
    plt.figure(figsize = (20, 15))
    sns.scatterplot(df[x], df[y]).set_title(title)
    plt.show()
plot_counts("DBNOs")
plot_counts("boosts")
plot_counts("headshotKills")
plot_counts("heals")
plot_counts("killStreaks")
plot_counts("kills")
plot_counts("revives")
plot_counts("roadKills")
plot_counts("teamKills")
plot_counts("vehicleDestroys")
plot_counts("weaponsAcquired")
plot_dists('walkDistance')
plot_counts("matchType")
df["headshot_rate"] = df["headshotKills"] / df["kills"]
df["headshot_rate"].fillna(0, inplace = True)
plot_dists("headshot_rate")
df["roadkills_rate"] = df["roadKills"] / df["kills"]
df["roadkills_rate"].fillna(0, inplace = True)
plot_dists("roadkills_rate")
plot_dists("killPlace")
plot_dists("swimDistance")
plot_dists("longestKill")
plot_scatters("rideDistance", "roadKills", "RoadKills by RideDistance")
plot_dists("killPlace")
plot_scatters("killPlace", "kills", "Kills by Killplace")
plot_scatters("winPlacePerc", "kills", "No of kills by winper")
df["matchDurationMinute"] = df["matchDuration"].apply(lambda x: x/60)
plot_scatters("winPlacePerc", "matchDurationMinute", "Wins by match duration")
plot_scatters("winPlacePerc","killPlace","")
plt.figure(figsize = (20, 15))
sns.pointplot(df["heals"], df["winPlacePerc"], linestyles="-")
sns.pointplot(df["boosts"], df["winPlacePerc"], color = "green", linestyles="--")
plt.xlabel("heals/boost")
plt.legend(["heals","boosts"]) 
plt.show()
plt.figure(figsize = (20, 15))
sns.pointplot(df["DBNOs"], df["assists"])
plt.grid()
plt.show()
plt.figure(figsize = (20, 15))
sns.pointplot(df["heals"], df["walkDistance"], linestyles="-")
sns.pointplot(df["boosts"], df["walkDistance"], color = "green", linestyles="--")
plt.xlabel("heals/boost")
plt.legend(["heals","boosts"]) 
plt.grid()
plt.show()
plt.figure(figsize = (20, 15))
sns.heatmap(df.corr(), annot = True, fmt='.1f')
plt.show()
del df
train_df = pd.read_csv("../input/train_V2.csv")
test_df = pd.read_csv("../input/test_V2.csv")
matchTyp = ['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo',
       'normal-squad-fpp', 'crashfpp', 'flaretpp', 'normal-solo-fpp',
       'flarefpp', 'normal-duo-fpp', 'normal-duo', 'normal-squad',
       'crashtpp', 'normal-solo']
mapping = {}
for i, j in enumerate(matchTyp):
    mapping[i] = j
train_df["matchTypeMap"] = train_df["matchType"].apply(lambda x: ''.join(str(i) for i, j in mapping.items() if x == j)).map(int64)
test_df["matchTypeMap"] = test_df["matchType"].apply(lambda x: ''.join(str(i) for i, j in mapping.items() if x == j)).map(int64)
train_df.drop(["matchType"], axis =1, inplace=True)
test_df.drop(["matchType"], axis =1, inplace=True)
train_df.dropna(inplace = True)
train_df.isnull().any().any()
X = train_df.drop(["Id", "groupId", "matchId", "winPlacePerc"], axis = 1)
y = train_df["winPlacePerc"]
test = test_df.drop(["Id", "groupId", "matchId"], axis = 1)
del train_df
model = xgboost.XGBRegressor(max_depth=17, gamma=0.3, learning_rate= 0.1)
model.fit(X,y)
del X
del y
xgboost.plot_importance(model)
pred = model.predict(test)
test_id = test_df["Id"]
submit_xg = pd.DataFrame({'Id': test_id, "winPlacePerc": pred} , columns=['Id', 'winPlacePerc'])
print(submit_xg.head())
submit_xg.to_csv("submission.csv", index = False)