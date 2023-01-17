# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")

#Check the shape of data
print(train.shape)
print(test.shape)
train.dtypes
train.isnull().sum()
train.winPlacePerc.fillna(train.winPlacePerc.mean(),inplace=True)
train.winPlacePerc.isnull().sum()
features = list(train.columns)
print(features)

#Based on player, so drop "Id", "groupId", "matchId", "winPlacePerc"
for i in ["Id","groupId","matchId","winPlacePerc"]:
    features.remove(i)
    
print(features)
kills = ["damageDealt", "DBNOs", "headshotKills", "killPlace", "killPoints", "kills", "killStreaks", "longestKill", "roadKills"]
teamwork = ["assists", "revives"]
med = ["boosts", "heals"]
movement = ["rideDistance", "swimDistance", "walkDistance"]
Others = ["maxPlace", "numGroups", "teamKills", "vehicleDestroys", "weaponsAcquired", "winPoints"]

print("kills: "+str(len(kills)))
print("teamwork: "+str(len(teamwork)))
print("med: "+str(len(med)))
print("movement: "+str(len(movement)))
print("Others: "+str(len(Others)))
data = train.copy()
for col in kills:
    print(data[col].describe().drop("count"))
    print()
data.loc[data.kills > 10,"kills"] = 10
data.loc[data.DBNOs > 10,"DBNOs"] = 10 
nrows=3
ncols=3
fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols*6,nrows*4))
for i in range(ncols):
    for j in range(nrows):
        idx=i*ncols+j
        sns.distplot(data[kills[idx]],kde=True,ax=axes[i][j])
sns.boxplot(x="kills",y="winPlacePerc",data=data)        
sns.boxplot(x="DBNOs",y="winPlacePerc",data=data)
sns.jointplot(x="winPlacePerc",y="damageDealt",data=data)
sns.distplot(data.longestKill,bins=100)
plt.scatter(x="winPlacePerc",y="longestKill",data=data)
data = train.copy()
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]
print(data["heals"].describe().drop("count"))
print(data["boosts"].describe().drop("count"))
for col in teamwork:
    print(data[col].describe().drop("count"))
    print()
team = train.groupby("groupId").mean()
print(team.shape)
team.head()
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,6))
sns.distplot(team[teamwork[0]],kde=False,ax=axes[0],bins=100)
sns.distplot(team[teamwork[1]],kde=False,ax=axes[1],bins=100)
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(18,6))
sns.boxplot(x=round(team.assists),y=team.winPlacePerc,ax=axes[0])
sns.boxplot(x=round(team.revives),y=team.winPlacePerc,ax=axes[1])
#Riding
ride = team.copy()
ride = ride[ride["rideDistance"]<ride["rideDistance"].quantile(0.99)]
print(ride["rideDistance"].describe().drop("count"))
sns.distplot(ride["rideDistance"])
sns.jointplot(x="winPlacePerc",y="rideDistance",data=team)
#Running
run = team.copy()
run = run[run["walkDistance"]<run["walkDistance"].quantile(0.99)]
print(run["walkDistance"].describe().drop("count"))
sns.distplot(run["walkDistance"])
sns.jointplot(x="winPlacePerc",y="walkDistance",data=team)
swim = team.copy()
swim = swim[swim["swimDistance"]<swim["swimDistance"].quantile(0.99)]
print(swim["swimDistance"].describe().drop("count"))
sns.distplot(swim["swimDistance"],kde=False)
swim['swimDistance'] = pd.cut(swim['swimDistance'], [-1, 0, 5, 20, 5286], labels=['0m','1-5m', '6-20m', '20m+'])
sns.boxplot(x="swimDistance",y="winPlacePerc",data=swim)