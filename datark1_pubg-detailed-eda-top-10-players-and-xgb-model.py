import numpy as np                    # linear algebra

import pandas as pd                   # database manipulation

import matplotlib.pyplot as plt       # plotting libraries

import seaborn as sns                 # nice graphs and plots

import warnings                       # libraries to deal with warnings

warnings.filterwarnings("ignore")



print("pandas version: {}".format(pd.__version__))

print("numpy version: {}".format(np.__version__))

print("seaborn version: {}".format(sns.__version__))
train = pd.read_csv('../input/train_V2.csv')



print('There are {:,} rows and {} columns in our dataset.'.format(train.shape[0],train.shape[1]))
train.head()
train.info()
train.describe()
missing_data = train.isna().sum().to_frame()

missing_data.columns=["Missing data"]
no_matches = train.loc[:,"matchId"].nunique()

print("There are {} matches registered in our database.".format(no_matches))
m_types = train.loc[:,"matchType"].value_counts().to_frame().reset_index()

m_types.columns = ["Type","Count"]

m_types
plt.figure(figsize=(15,8))

ticks = m_types.Type.values

ax = sns.barplot(x="Type", y="Count", data=m_types)

ax.set_xticklabels(ticks, rotation=60, fontsize=14)

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



wedges, texts, autotexts = ax1.pie(aggregated_mt["count"],textprops=dict(color="w"), autopct='%1.1f%%', startangle=90)



ax1.axis('equal')

ax1.legend(wedges, labels,

          title="Types",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=12, weight="bold")

plt.show()
plt.figure(figsize=(15,8))

ax = sns.distplot(train["numGroups"])

ax.set_title("Number of groups")

plt.show()
plt.figure(figsize=(15,8))

ax1 = sns.boxplot(x="kills",y="damageDealt", data = train)

ax1.set_title("Damage Dealt vs. Number of Kills")

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
plt.figure(figsize=(15,8))

ax3 = sns.boxplot(x="killStreaks",y="kills", data = train)

ax3.set_title("Number of kill streaks vs. Number of Kills")

plt.show()
dist = train[train['longestKill']<200]

plt.rcParams['axes.axisbelow'] = True

dist.hist('longestKill', bins=20, figsize = (16,8))

plt.show()
print("Average longest kill distance a player achieve is {:.1f}m, 95% of them not more than {:.1f}m and a maximum distance is {:.1f}m." .format(train['longestKill'].mean(),train['longestKill'].quantile(0.95),train['longestKill'].max()))
walk0 = train["walkDistance"] == 0

ride0 = train["rideDistance"] == 0

swim0 = train["swimDistance"] == 0

print("{} of players didn't walk at all, {} players didn't drive and {} didn't swim." .format(walk0.sum(),ride0.sum(),swim0.sum()))
walk0_rows = train[walk0]

print("Average place of non-walking players is {:.3f}, minimum is {} and the best is {}, 95% of players has a score below {}." 

      .format(walk0_rows["winPlacePerc"].mean(), walk0_rows["winPlacePerc"].min(), walk0_rows["winPlacePerc"].max(),walk0_rows["winPlacePerc"].quantile(0.95)))

walk0_rows.hist('winPlacePerc', bins=40, figsize = (16,8))

plt.show()
suspects = train.query('winPlacePerc ==1 & walkDistance ==0').head()

suspects.head()
print("Maximum ride distance for suspected entries is {:.3f} meters, and swim distance is {:.1f} meters." .format(suspects["rideDistance"].max(), suspects["swimDistance"].max()))
ride = train.query('rideDistance >0 & rideDistance <10000')

walk = train.query('walkDistance >0 & walkDistance <4000')

ride.hist('rideDistance', bins=40, figsize = (15,10))

walk.hist('walkDistance', bins=40, figsize = (15,10))

plt.show()
travel_dist = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]

travel_dist = travel_dist[travel_dist<5000]

travel_dist.hist(bins=40, figsize = (15,10))

plt.show()
print("Average number of acquired weapons is {:.3f}, minimum is {} and the maximum {}, 99% of players acquired less than weapons {}." 

      .format(train["weaponsAcquired"].mean(), train["weaponsAcquired"].min(), train["weaponsAcquired"].max(), train["weaponsAcquired"].quantile(0.99)))

train.hist('weaponsAcquired', figsize = (20,10),range=(0, 10), align="left", rwidth=0.9)

plt.show()
ax = sns.clustermap(train.corr(), annot=True, linewidths=.6, fmt= '.2f', figsize=(20, 15))

plt.show()
top10 = train[train["winPlacePerc"]>0.9]

print("TOP 10% overview\n")

print("Average number of kills: {:.1f}\nMinimum: {}\nThe best: {}\n95% of players within: {} kills." 

      .format(top10["kills"].mean(), top10["kills"].min(), top10["kills"].max(),top10["kills"].quantile(0.95)))
plt.figure(figsize=(15,8))

ax3 = sns.boxplot(x="DBNOs",y="kills", data = top10)

ax3.set_title("NUmber of DBNOs vs. Number of Kills")

plt.show()
fig, ax1 = plt.subplots(figsize = (15,10))

walk.hist('walkDistance', bins=40, figsize = (15,10), ax = ax1)

walk10 = top10[top10['walkDistance']<5000]

walk10.hist('walkDistance', bins=40, figsize = (15,10), ax = ax1)



print("Average walking distance: " + str(top10['walkDistance'].mean()))
fig, ax1 = plt.subplots(figsize = (15,10))

ride.hist('rideDistance', bins=40, figsize = (15,10), ax = ax1)

ride10 = top10.query('rideDistance >0 & rideDistance <10000')

ride10.hist('rideDistance', bins=40, figsize = (15,10), ax = ax1)

print("Average riding distance: " + str(top10['rideDistance'].mean()))
print("On average the best 10% of players have the longest kill at {:.3f} meters, and the best score is {:.1f} meters." .format(top10["longestKill"].mean(), top10["longestKill"].max()))
ax = sns.clustermap(top10.corr(), annot=True, linewidths=.5, fmt= '.2f', figsize=(20, 15))

plt.show()
import xgboost as xgb



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer



train.dropna(subset=["winPlacePerc"], inplace=True) # droping rows with missing labels



X = train.drop(["Id","groupId","matchId","matchType","winPlacePerc"], axis=1)

y = train["winPlacePerc"]



col_names = X.columns



transformer = Normalizer().fit(X)

X = transformer.transform(X)
X = pd.DataFrame(X, columns=col_names)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)



D_train = xgb.DMatrix(X_train, label=Y_train)

D_test = xgb.DMatrix(X_test, label=Y_test)
param = {

    'eta': 0.15, 

    'max_depth': 5,  

    'num_class': 2} 



steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)
fig, ax1 = plt.subplots(figsize=(8,15))

xgb.plot_importance(model, ax=ax1)

plt.show()
from sklearn.metrics import mean_squared_error



preds = model.predict(D_test)

best_preds = np.asarray([np.argmax(line) for line in preds])



print("MSE = {}".format(mean_squared_error(Y_test, best_preds)))