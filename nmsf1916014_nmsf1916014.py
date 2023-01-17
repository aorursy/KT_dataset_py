import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# import training data

train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")



train.head()
train.info()
# check how many rows having missing data

train.isnull().sum()
# take a look at this row

train[train['winPlacePerc'].isnull()]
# look like an incorrect record

# remove this row

train.drop(2744604, inplace=True)
train['kills'].describe()
print('The average number of kills is {:.4f}. 99% players have {} kills or less. The most kills ever is {:.0f}.'

      .format(train['kills'].mean(), train['kills'].quantile(0.99), train['kills'].max()))
data = train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(15, 10))

sns.countplot(data['kills'].astype('str').sort_values())

plt.title('Kill Count', fontsize=20)
data = train.copy()

data = data[data['kills']==0]

plt.figure(figsize=(15, 10))

sns.distplot(data['damageDealt'])

plt.title('Damage dealt by Player who killed 0 enemy')
data['damageDealt'].describe()
print('{:.0f} zero-killers ({:.4%}) won the game.'

     .format(len(data[data['winPlacePerc']==1]), len(data[data['winPlacePerc']==1])/len(data)))



print('{:.0f} zero-killers ({:.4%}) won the game with 0 damage dealt.'

     .format(len(data[(data['winPlacePerc']==1) & (data['damageDealt']==0)]),

             len(data[(data['winPlacePerc']==1) & (data['damageDealt']==0)])/len(data)))
sns.jointplot(data=train, x='winPlacePerc', y='kills', size=10, ratio=3)
kills = train.copy()



kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 100], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])



plt.figure(figsize=(15,8))

sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
train['walkDistance'].describe()
print('The average walking distance is {:.1f}m. 99% players walked {:.1f}m or less. The max distance is {:.1f}m.'

     .format(train['walkDistance'].mean(), train['walkDistance'].quantile(0.99), train['walkDistance'].max()))
data = train.copy()

plt.figure(figsize=(15, 10))

sns.distplot(data['walkDistance'])

plt.title('Walking Distance')
print('{:.0f} ({:.4%}) players walked 0 meter. That means they died just reached the ground or afk.'

     .format(len(data[data['walkDistance']==0]), len(data[data['walkDistance']==0])/len(data)))
sns.jointplot(data=train, x='winPlacePerc', y='walkDistance', size=10, ratio=3)
print('The average number of heal items used is {:.1f}. 99% players used {:.1f} or less. Max number of used heal items is {:.0f}.'

     .format(train['heals'].mean(), train['heals'].quantile(0.99), train['heals'].max()))

print('')

print('The average number of boost items used is {:.1f}. 99% players used {:.1f} or less. Max number of used boost items is {:.0f}.'

     .format(train['boosts'].mean(), train['boosts'].quantile(0.99), train['boosts'].max()))
train[['heals', 'boosts']].describe()
# this will take a few minutes



# data = train.copy()



# f, ax1 = plt.subplots(figsize=(20, 10))

# l1 = sns.pointplot(data=data, x='heals', y='winPlacePerc', color='lime', alpha=0.8)

# l2 = sns.pointplot(data=data, x='boosts', y='winPlacePerc', color='blue', alpha=0.8)

# plt.legend([l1, l2], ['heals', 'boosts'], loc='lower center')

# plt.xlabel('Number of heal/boost items', fontsize=20, color='blue')

# plt.ylabel('Win Percentage', fontsize=20, color='blue')

# plt.title('Heals vs Boosts', fontsize=20, color='blue')

# plt.grid()
sns.jointplot(data=data, x='winPlacePerc', y='heals', size=10, ratio=3, color='lime')
sns.jointplot(data=data, x='winPlacePerc', y='boosts', size=10, ratio=3)
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train.corr(), annot=True, linewidth=0.5, fmt='.1f')
# top 5 most positive correlated variables

k = 6

f, ax = plt.subplots(figsize=(10, 10))

cols = train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index

cm = train[cols].corr()

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
train['num_players'] = train.groupby('matchId')['matchId'].transform('count')
data = train.copy()

data = data[data['num_players']>=75]

plt.figure(figsize=(15, 10))

sns.countplot(data['num_players'])

plt.title('Number of Players', fontsize=20)
# normalize kills, damageDealt, maxPlace, matchDuration

train['killsNorm'] = train['kills'] * ((100 - train['num_players']) / 100 + 1)

train['damageDealtNorm'] = train['damageDealt'] * ((100 - train['num_players']) / 100 + 1)

train['maxPlaceNorm'] = train['maxPlace'] * ((100 - train['num_players']) / 100 + 1)

train['matchDurationNorm'] = train['matchDuration'] * ((100 - train['num_players']) / 100 + 1)



train[['Id', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm',

       'matchDuration', 'matchDurationNorm']].head()
train['healsAndBoosts'] = train['heals'] + train['boosts']

train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
# find out players who are getting kills without moving

train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
# create feature headshot_rate

train['headshot_rate'] = train['headshotKills'] / train['kills']

train['headshot_rate'] = train['headshot_rate'].fillna(0)
display(train[train['killsWithoutMoving'] == True].shape)

train[train['killsWithoutMoving'] == True].head(10)
# remove outliers

train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train['roadKills'].describe()
train['roadKills'].value_counts()
# remove players with more than 10 roadKills

train.drop(train[train['roadKills'] > 10].index, inplace=True)
plt.figure(figsize=(15, 5))

sns.countplot(data=train, x='kills')

plt.title('Kills', fontsize=20)
# players who got more than 30 kills

display(train[train['kills'] > 30].shape)

train[train['kills'] > 30].head(10)
# remove outliers

train.drop(train[train['kills'] > 30].index, inplace=True)
plt.figure(figsize=(15, 5))

sns.distplot(train['headshot_rate'], bins=10)
display(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)

train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head(10)
plt.figure(figsize=(15, 5))

sns.distplot(train['longestKill'], bins=10)
display(train[train['longestKill'] >= 1000].shape)

train[train['longestKill'] >= 1000].head(10)
# remove outliers

train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()
# walkDistance

plt.figure(figsize=(15, 5))

sns.distplot(train['walkDistance'], bins=10)
display(train[train['walkDistance'] >= 10000].shape)

train[train['walkDistance'] >= 10000].head(10)
# remove outliers

train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
# rideDistance

plt.figure(figsize=(15, 5))

sns.distplot(train['rideDistance'], bins=10)
display(train[train['rideDistance'] >= 20000].shape)

train[train['rideDistance'] >= 20000].head(10)
# remove outliers

train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
# swimDistance

plt.figure(figsize=(15, 5))

sns.distplot(train['swimDistance'], bins=10)
display(train[train['swimDistance'] >= 2000].shape)

train[train['swimDistance'] >= 2000].head(10)
# remove outliers

train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
plt.figure(figsize=(15, 5))

sns.distplot(train['weaponsAcquired'], bins=100)
display(train[train['weaponsAcquired'] >= 80].shape)

train[train['weaponsAcquired'] >= 80].head(10)
# remove outliers

train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
plt.figure(figsize=(15, 5))

sns.distplot(train['heals'], bins=10)
display(train[train['heals'] >= 40].shape)

train[train['heals'] >= 40].head(10)
# remove outliers

train.drop(train[train['heals'] >= 40].index, inplace=True)
print('There are {} different Match types in the dataset.'

     .format(train['matchType'].nunique()))
# one hot encode matchType

train = pd.get_dummies(train, columns=['matchType'])



# take a look

matchType_encoding = train.filter(regex='matchType')

matchType_encoding.head()
# turn groupId, matchId into categorical types

train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')



# get category coding for groupId, matchId

train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes



# get rid of old columns

train.drop(columns=['groupId', 'matchId'], inplace=True)



train[['groupId_cat', 'matchId_cat']].head()
# drop Id column

train.drop(columns=['Id'], inplace=True)
from sklearn.model_selection import train_test_split
X_train = train.drop(columns=['winPlacePerc'])

y_train = train['winPlacePerc']



del train
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
print('X_train shape:', X_train.shape)

print('y_train shape:', y_train.shape)

print('X_test shape:', X_test.shape)

print('y_test shape:', y_test.shape)
from sklearn.metrics import mean_absolute_error



def print_score(m: RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train), y_train),

           'mae val: ', mean_absolute_error(m.predict(X_test), y_test)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
# base model

m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m1.fit(X_train, y_train)

print_score(m1)
# top 10 most predictive features

importances = pd.DataFrame(m1.feature_importances_, index=X_train.columns,

                           columns=['importance']).sort_values('importance', ascending=False)

importances = importances.reset_index()



importances[:10]
plt.figure(figsize=(20, 10))

sns.barplot(data=importances[:20], x='importance', y='index')
# keep significant features

to_keep = importances[importances['importance'] >= 0.005]

print('Significant features:', len(to_keep))

to_keep
X_train = X_train[to_keep['index']]

X_test = X_test[to_keep['index']]
# train with significant features only

m2 = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m2.fit(X_train, y_train)

print_score(m2)
importances_to_keep = pd.DataFrame(m2.feature_importances_, index=to_keep['index'],

                           columns=['importance']).sort_values('importance', ascending=False)

importances_to_keep = importances_to_keep.reset_index()



importances_to_keep[:10]
plt.figure(figsize=(20, 10))

sns.barplot(data=importances_to_keep, x='importance', y='index')
from scipy.cluster import hierarchy as hc

from scipy.stats import spearmanr
# Dendrogram

corr = np.round(spearmanr(X_train).correlation, 4)

corr_condensed = hc.distance.squareform(1 - corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(15, 10))

dendrogram = hc.dendrogram(z, labels=X_train.columns, orientation='left', leaf_font_size=15)
# correlation heatmap

corr = X_train.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(corr)
# apply feature engineering to the test set

test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

test['headshot_rate'] = test['headshotKills'] / test['kills']

test['headshot_rate'] = test['headshot_rate'].fillna(0)

test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

test['num_players'] = test.groupby('matchId')['matchId'].transform('count')

# normalize

test['killsNorm'] = test['kills']*((100-test['num_players'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['num_players'])/100 + 1)

test['maxPlaceNorm'] = test['maxPlace']*((100-test['num_players'])/100 + 1)

test['matchDurationNorm'] = test['matchDuration']*((100-test['num_players'])/100 + 1)

test['healsAndBoosts'] = test['heals'] + test['boosts']

test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))



# turn groupId, matchId into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes



# Remove irrelevant features from the test set

test_pred = test[to_keep['index']].copy()



# Fill NaN with 0 (temporary)

test_pred.fillna(0, inplace=True)

test_pred.head()
# submission

predictions = np.clip(a=m2.predict(test_pred), a_min=0.0, a_max=1.0)

pred_df = pd.DataFrame({'Id': test['Id'], 'winPlacePerc': predictions})



pred_df.to_csv('submission.csv', index=False)