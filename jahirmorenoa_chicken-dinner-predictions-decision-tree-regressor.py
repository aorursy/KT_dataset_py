import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_pubg= pd.read_csv('../input/train.csv') 
train_pubg.info()
sample_pubg = train_pubg[0:1000000]
plt.figure(figsize=(15,12))
sns.heatmap(sample_pubg.corr(), cmap='viridis', annot=True)
plt.title('Correlation Matrix for the sample')
plt.tight_layout()
plt.figure(figsize=(14,10))
sns.scatterplot(x='kills',y='damageDealt', hue ='DBNOs', data=sample_pubg[sample_pubg['numGroups']<50])
plt.title('Squads and Duos only')
sns.scatterplot(sample_pubg['damageDealt'], sample_pubg['DBNOs'])
sample_kills = sample_pubg[['killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill','headshotKills','roadKills', 'teamKills','winPlacePerc']]
plt.figure(figsize=(15,4))
plt.subplot(141)
sns.scatterplot(sample_kills['killPoints'], sample_kills['headshotKills'], alpha=0.5, color='red')
plt.subplot(142)
sns.scatterplot(sample_kills['killPoints'], sample_kills['kills'], alpha = 0.5, color='blue')
plt.subplot(143)
sns.scatterplot(sample_kills['killPoints'], sample_kills['longestKill'], alpha = 0.5, color='purple')
plt.subplot(144)
sns.scatterplot(sample_kills['killPoints'], sample_kills['winPlacePerc'], alpha = 0.5, color='green')
plt.tight_layout()
plt.figure(figsize=(14,10))
sns.heatmap(sample_kills.corr(), cmap='viridis', annot=True, linewidths=1)
plt.title('Kills Correlation Matrix')
from sklearn.model_selection import train_test_split
X=train_pubg[[ 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']]
y=train_pubg['winPlacePerc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeRegressor
my_treemodel = DecisionTreeRegressor()
my_treemodel.fit(X_train, y_train)
predictions_treemodel = my_treemodel.predict(X_test)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predictions_treemodel))
