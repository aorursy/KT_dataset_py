# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
data.head()
len(data)
data['blueWins'].value_counts()
# Wins are almost he same number, so we have an equal number of both wins
print('Mean of blue wards places when blue team wins : %.5f' %data[data.blueWins == 1]['blueWardsPlaced'].mean())
print('Mean of blue wards places when blue team wins : %.5f' %data[data.blueWins == 0]['blueWardsPlaced'].mean())
# Victory doesn't depend on the wards placed but on the difference between the two teams
data['blueWardsPlacedDiff'] = data['blueWardsPlaced'] - data['redWardsPlaced']
print('Mean difference between wards places when blue team wins : %.5f' %data[data.blueWins == 1]['blueWardsPlacedDiff'].mean())
print('Mean difference between wards places when blue team loses : %.5f' %data[data.blueWins == 0]['blueWardsPlacedDiff'].mean())
# More chance to win when you placed more wards but not a great difference
# Chance to win on wards placed difference

data['blueWardsPlacedDiffBins'] = pd.qcut(data['blueWardsPlacedDiff'], q=10)
prob_wins = data.groupby('blueWardsPlacedDiffBins')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueWardsPlacedDiffBins', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueWardsPlacedDiffBins", y="win_probability", data=prob_wins)

l = ax.set(ylim=(0.4,0.55))
# Not a big difference

# More probability to win with a positive difference but difference in [23, 234] has more probability for a lose
print('Mean of blue wards destroyed when blue team wins : %.5f' %data[data.blueWins == 1]['blueWardsDestroyed'].mean())
print('Mean of blue wards destroyed  when blue team loses : %.5f' %data[data.blueWins == 0]['blueWardsDestroyed'].mean())
# A little more wards destroyed by the winning team but let's see the difference to understand more the win probability
data['blueWardsDestroyedDiff'] = data['blueWardsDestroyed'] - data['redWardsDestroyed']
print('Mean difference between wards destroyed when blue team wins : %.5f' %data[data.blueWins == 1]['blueWardsDestroyedDiff'].mean())
print('Mean difference between wards destroyed when blue team loses : %.5f' %data[data.blueWins == 0]['blueWardsDestroyedDiff'].mean())
# As for the woards place, more chance to win when you destroyed more wards but not a great difference
# Chance to win on wards destroyed difference

prob_wins = data.groupby('blueWardsDestroyedDiff')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueWardsDestroyedDiff', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueWardsDestroyedDiff", y="win_probability", data=prob_wins)
# There don't seem to be a lot of correlation
prob_wins = data.groupby('blueFirstBlood')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueFirstBlood', 'win_probability']
plt.figure(figsize=(8,5))

ax = sns.barplot(x="blueFirstBlood", y="win_probability", data=prob_wins)
# Having the first blood seems to be a good advantage
print('Mean of blue kills when blue team wins : %.5f' %data[data.blueWins == 1]['blueKills'].mean())
print('Mean of blue kills when blue team loses : %.5f' %data[data.blueWins == 0]['blueKills'].mean())
# More kills for the winning teams
data['blueKillsDiff'] = data['blueKills'] - data['redKills']
print('Mean difference between kills when blue team wins : %.5f' %data[data.blueWins == 1]['blueKillsDiff'].mean())
print('Mean difference between kills when blue team loses : %.5f' %data[data.blueWins == 0]['blueKillsDiff'].mean())
# Chance to win on kills difference

prob_wins = data.groupby('blueKillsDiff')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueKillsDiff', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueKillsDiff", y="win_probability", data=prob_wins)
# Ok, now it's clear (80% to win with 5+ difference of kills)
# Deaths is the opposite as kills

# It will not be useful when creating the model (death = - kills)
data['blueAssistsDiff'] = data['blueAssists'] - data['redAssists']
# Chance to win on assists difference

prob_wins = data.groupby('blueAssistsDiff')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueAssistsDiff', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueAssistsDiff", y="win_probability", data=prob_wins)
# Almost same results as kills
print('Mean of elite monsters killed by blue team when blue team wins : %.5f' %data[data.blueWins == 1]['blueEliteMonsters'].mean())
print('Mean of elite monsters killed by blue team  when blue team wins: %.5f' %data[data.blueWins == 0]['blueEliteMonsters'].mean())
# More Elite Monsters killed for the winning team
data['blueEliteMonstersDiff'] = data['blueEliteMonsters'] - data['redEliteMonsters']
# Chance to win on elite monsters killed difference

prob_wins = data.groupby('blueEliteMonstersDiff')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueEliteMonstersDiff', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueEliteMonstersDiff", y="win_probability", data=prob_wins)
# Killing Elite Monsters seems to be an advantage
# Elite Monsters contains Dragons and Heralds so we will just create the dragons killed difference for later

data['blueDragonsDiff'] = data['blueDragons'] - data['redDragons']
# Elite Monsters contains Dragons and Heralds so we will just create the dragons killed difference for later

data['blueHeraldsDiff'] = data['blueHeralds'] - data['redHeralds']
print('Mean of towers destroyed by blue team when blue team wins : %.5f' %data[data.blueWins == 1]['blueTowersDestroyed'].mean())
print('Mean of towers destroyed by blue team  when blue team wins: %.5f' %data[data.blueWins == 0]['blueTowersDestroyed'].mean())
# A little more tower destoyed for the winning team, the difference is explained by the few number of tower
sns.countplot(data['blueTowersDestroyed'])
# A majority of no towers destroyed
data['blueTowersDestroyedDiff'] = data['blueTowersDestroyed'] - data['redTowersDestroyed']
# Chance to win on towers destroyed difference

prob_wins = data.groupby('blueTowersDestroyedDiff')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueTowersDestroyedDiff', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueTowersDestroyedDiff", y="win_probability", data=prob_wins)
# Having more than 1 towers destroyed difference is a win
# We will observe three columns as they are about the same aspect
print('Mean of blue total gold when blue team wins : %.5f' %data[data.blueWins == 1]['blueTotalGold'].mean())
print('Mean of blue total gold when blue team loses : %.5f' %data[data.blueWins == 0]['blueTotalGold'].mean())
# More gold for the winning team (because more kills ? We will see this later)
# Chance to win on gold difference

data['blueGoldDiffBins'] = pd.qcut(data['blueGoldDiff'], q=10, duplicates='drop')
prob_wins = data.groupby('blueGoldDiffBins')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueGoldDiffBins', 'win_probability']
plt.figure(figsize=(20,8))

ax = sns.barplot(x="blueGoldDiffBins", y="win_probability", data=prob_wins)
# Difference in gold seems to be correlated to win
data['blueGoldPerMinDiff'] = data['blueGoldPerMin'] - data['redGoldPerMin']
# Gold Per Min is equivalent to Total Gold
print('Mean of blue average level when blue team loses : %.5f' %data[data.blueWins == 1]['blueAvgLevel'].mean())
print('Mean of blue average level when blue team loses : %.5f' %data[data.blueWins == 0]['blueAvgLevel'].mean())
# Mean is quite the same
data['blueAvgLevelDiff'] = data['blueAvgLevel'] - data['redAvgLevel']
# Chance to win on average level difference

data['blueAvgLevelDiffBins'] = pd.qcut(data['blueAvgLevelDiff'], q=10, duplicates='drop')
prob_wins = data.groupby('blueAvgLevelDiffBins')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueAvgLevelDiffBins', 'win_probability']
plt.figure(figsize=(20,8))

ax = sns.barplot(x="blueAvgLevelDiffBins", y="win_probability", data=prob_wins)
# Some correlation between level average difference and win
print('Mean of blue total experience when blue team loses : %.5f' %data[data.blueWins == 1]['blueTotalExperience'].mean())
print('Mean of blue total experience when blue team loses : %.5f' %data[data.blueWins == 0]['blueTotalExperience'].mean())
# More Experience for winning team (because of more kills ?)
# Chance to win on experience difference

data['blueTotalExperienceDiffBins'] = pd.qcut(data['blueExperienceDiff'], q=10, duplicates='drop')
prob_wins = data.groupby('blueTotalExperienceDiffBins')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueTotalExperienceDiffBins', 'win_probability']
plt.figure(figsize=(20,8))

ax = sns.barplot(x="blueTotalExperienceDiffBins", y="win_probability", data=prob_wins)
# Good correlation between experience and win
print('Mean of blue total minions killed when blue team loses : %.5f' %data[data.blueWins == 1]['blueTotalMinionsKilled'].mean())
print('Mean of blue total minions killed when blue team loses : %.5f' %data[data.blueWins == 0]['blueTotalMinionsKilled'].mean())
# More minions killed for winning teams
data['blueTotalMinionsKilledDiff'] = data['blueTotalMinionsKilled'] - data['redTotalMinionsKilled']
# Chance to win on wards placed difference

data['blueTotalMinionsKilledDiffBins'] = pd.qcut(data['blueTotalMinionsKilledDiff'], q=10, duplicates='drop')
prob_wins = data.groupby('blueTotalMinionsKilledDiffBins')['blueWins'].mean()

prob_wins = prob_wins.reset_index()

prob_wins.columns = ['blueTotalMinionsKilledDiffBins', 'win_probability']
plt.figure(figsize=(15,8))

ax = sns.barplot(x="blueTotalMinionsKilledDiffBins", y="win_probability", data=prob_wins)
# Some correlation between total minions killed difference and win
# Jungle Minions included in Minions

data['blueTotalJungleMinionsKilledDiff'] = data['blueTotalJungleMinionsKilled'] - data['redTotalJungleMinionsKilled']
# CS is equivalent to Minions Killed

data['blueCSPerMinDiff'] = data['blueCSPerMin'] - data['redCSPerMin']
# Removing columns with 'red' for the heatmap

cols = [c for c in data.columns if c.lower()[:3] != 'red']

data_blue = data[cols]



# calculate the correlation matrix

data_corr = data_blue.corr()['blueWins']
# Removing columns with duplicate correlation

data_blue = data_blue.drop(['blueGoldPerMinDiff', 'blueGoldPerMin', 'blueCSPerMinDiff', 'blueTotalMinionsKilled'], 1)
# Get columns with at least 0.2 correlation

data_corr = data_blue.corr()['blueWins']

cols = data_corr[abs(data_corr) > 0.2].index.tolist()

data_blue = data_blue[cols]
# plot the heatmap

data_corr = data_blue.corr()

plt.figure(figsize=(10,8))

sns.heatmap(data_corr, 

        xticklabels=data_corr.columns,

        yticklabels=data_corr.columns, cmap=sns.diverging_palette(220, 20, n=200))
data_blue.corr()['blueWins'].sort_values(ascending=False)
# The biggest factor is the gold difference followed by the experience difference and kills difference
data_blue['blueDragons'].corr(data_blue['blueEliteMonsters'])
data_blue['blueDragonsDiff'].corr(data_blue['blueEliteMonstersDiff'])
data_blue['blueEliteMonsters'].corr(data_blue['blueEliteMonstersDiff'])
# Remove blueDragons & blueEliteMonsters (correlation 0.78), blueDragonsDiff (0.83 correlation with blueEliteMonstersDiff)

data_blue = data_blue.drop(['blueDragons', 'blueEliteMonsters', 'blueDragonsDiff'], 1)
data_blue['blueExperienceDiff'].corr(data_blue['blueAvgLevel'])
data['blueKills'].corr(data['blueTotalGold'])
data_blue['blueKillsDiff'].corr(data_blue['blueAssistsDiff'])
# Remove blueKills (correlation 0.89 with blueTotalGold) & blueAssistsDiff(correlation 0.83 with blueAssistsDiff)

data_blue = data_blue.drop(['blueKills', 'blueAssistsDiff'], 1)
data_blue.columns
# Output is binary so 0 or 1

# We will test Logistic Regression, Decision Tree, Random Forest, KNeighbors
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
X = data_blue.drop("blueWins", axis=1)

Y = data_blue["blueWins"]
# Normalize features columns

# Models performe better when values are close to normally distributed

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)
# Split 20% test, 80% train



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression()

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_test)

acc_log = accuracy_score(Y_pred_log, Y_test)

acc_log
t = tree.DecisionTreeClassifier()



# search the best params

grid = {'min_samples_split': [5, 10, 20, 50, 100]},



clf_tree = GridSearchCV(t, grid, cv=10)

clf_tree.fit(X_train, Y_train)



Y_pred_tree = clf_tree.predict(X_test)



# get the accuracy score

acc_tree = accuracy_score(Y_pred_tree, Y_test)

print(acc_tree)
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': range(2,10,2)}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_test)

print(acc_rf)
# KNN or k-Nearest Neighbors



knn = KNeighborsClassifier()



# search the best params

grid = {"n_neighbors":np.arange(1,100)}

clf_knn = GridSearchCV(knn, grid, cv=10)

clf_knn.fit(X_train,Y_train) 



# get accuracy score

Y_pred_knn = clf_knn.predict(X_test) 

acc_knn = accuracy_score(Y_pred_knn, Y_test)

print(acc_knn)
# The logistic regression seems to be the best model