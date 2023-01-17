# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head()
train.isnull().sum()
print(train.info())
train = train.dropna(how = 'any')
plt.figure(figsize = (8, 8))
plt.scatter(range(train.shape[0]), np.sort(train["winPlacePerc"]))
plt.xlabel('Index Range')
plt.ylabel('Target Value')
plt.title('Check for outliers')
plt.show()
print(train["winPlacePerc"].max())
print(train["winPlacePerc"].min())
plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
sns.distplot(train["winPlacePerc"],bins = 100)
plt.xlabel("Win Palce Percent")
plt.title("Target Value distribution")
plt.subplot(1, 2, 2)
sns.distplot(np.log(train["winPlacePerc"][train["winPlacePerc"]>0.0]),bins = 100)
plt.xlabel("Win Palce Percent")
plt.title("Target Value distribution")
plt.show()
print("Once a player got too enthusiastic and killed {} people".format(train["kills"].max()))
plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
sns.distplot(train["kills"])
plt.xlabel("kills")
plt.title("Kills")
plt.subplot(1, 2, 2)
plt.scatter(train["winPlacePerc"], train["kills"])
plt.xlabel("Target Win Perc")
plt.ylabel("Kills")
plt.title("Win Perc v/s Kills")
plt.show()
plt.figure(figsize=(8, 8))
plt.scatter(train["winPlacePerc"], train["walkDistance"])
plt.xlabel("Target Win Perc")
plt.ylabel("Distance Walked in that game")
plt.title("Win Perc v/s Distance Walked")
plt.show()
plt.figure(figsize=(8, 8))
plt.scatter(train["winPlacePerc"], train["matchDuration"])
plt.xlabel("Target Win Perc")
plt.ylabel("Match Duration")
plt.title("Win Perc v/s Match Duration")
plt.show()
plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
sns.distplot(train["rideDistance"])
plt.xlabel("rideDistance")
plt.title("Distance covered on vehicles")
plt.subplot(1, 2, 2)
plt.scatter(train["winPlacePerc"], train["rideDistance"])
plt.xlabel("Win Percentile")
plt.ylabel("Distance travelled on vehicle")
plt.title("Win Percentile v/s rideDistance")
plt.show()
plt.figure(figsize = (8, 8))
plt.scatter(train["winPlacePerc"], train["damageDealt"])
plt.xlabel("Win Percentile")
plt.ylabel("Damage Dealt")
plt.title("Win Percentile v/s damageDealt")
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=train)
plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')
plt.grid()
plt.show()
plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
plt.scatter(train["winPlacePerc"], train["boosts"])
plt.xlabel("Win Percentile")
plt.ylabel("Boosts Used")
plt.title("Win Percentile v/s boosts")
plt.subplot(1, 2, 2)
plt.scatter(train["winPlacePerc"], train["heals"], color = 'red')
plt.xlabel("Win Percentile")
plt.ylabel("Number of healing items used.")
plt.title("Win Percentile v/s heals")
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# g = sns.pairplot(train, vars = ["winPlacePerc", "kills", "walkDistance", "matchDuration"])
# plt.show()
plt.figure(figsize = (12, 8))
sns.violinplot(x = 'matchType', y = 'winPlacePerc', data = train)
plt.xticks(rotation = 'vertical')
plt.show()
train["totalDistance"] = train["rideDistance"] + train["swimDistance"] + train["walkDistance"]
test["totalDistance"] = test["rideDistance"] + test["swimDistance"] + test["walkDistance"]
train["totalKills"] = train["roadKills"] + train["kills"] +train["headshotKills"]
test["totalKills"] = test["roadKills"] + train["kills"] +test["headshotKills"]
y = train["winPlacePerc"]
train.drop(["winPlacePerc", "matchType", "groupId", "matchId", "Id", "winPoints", "rankPoints",
            "walkDistance", "swimDistance", "rideDistance", "headshotKills", "kills", "roadKills"], inplace = True, axis = 1)
id = test["Id"]
test.drop(["matchType", "groupId", "matchId", "Id", "winPoints", "rankPoints", "walkDistance",
           "swimDistance", "rideDistance", "headshotKills", "kills", "roadKills"], inplace = True, axis = 1)
train.info()
params = {
    'num_leaves': 144,
    'learning_rate': 0.1,
    'n_estimators': 800,
    'max_depth':12,
    'max_bin':55,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'feature_fraction':0.9,
    'verbose':50, 
    'early_stopping_rounds':100,
    'metric':'mae'}
n_estimators = 3000
x_train, x_valid, y_train, y_valid = train_test_split(train, y, test_size=0.10)
d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)
watchlist = [d_valid]

model = lgb.train(params, d_train, n_estimators, verbose_eval=50, valid_sets=[d_train, d_valid])

preds = model.predict(test)
preds = np.clip(a = preds, a_min = 0.0, a_max = 1.0)
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(),
                   'split':model.feature_importance('split'),
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp)
sub = pd.DataFrame({"Id":id, "winPlacePerc":preds})
sub.to_csv('lightgbm_benchmark.csv', index = False)
sub.head()