#Import Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import gc

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from mlxtend.evaluate import feature_importance_permutation

from sklearn.model_selection import GridSearchCV

import gc

import os

import sys

#Figures Inline and Visualization style

%matplotlib inline

sb.set()
train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')
train.head()
train['matchType'].value_counts().plot.bar(title='Match Type VS Count', figsize=(14,6))
train.count()
train.dropna(inplace=True)
test.count()
train.describe()
f,ax = plt.subplots(figsize=(20, 20))

sb.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=sb.color_palette("RdBu", 20))

ax.set_xlabel('variables', size=14, color="black")

ax.set_ylabel('variables', size=14, color="black")

ax.set_title('Correlation Matrix', size=18, color="black")

plt.show()
print("In the average, a person kills {:.4f} players, About 99% players have a kill count less than or equal  to {}, while the most kills ever recorded is {}.".format(train['kills'].mean(),train['kills'].quantile(0.99), train['kills'].max()))
data = train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(15,8))

sb.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count",fontsize=15)

plt.show()
data = train[['kills','winPlacePerc']].copy()

data.loc[data['kills'] > data['kills'].quantile(0.99), 'kills'] = '8+'

order = data.groupby('kills').mean().reset_index()['kills']

fig, ax = plt.subplots(figsize=(20,8))

a = sb.boxplot(x='kills', y='winPlacePerc', data=data, ax=ax, color="#2196F3", order=order)

ax.set_xlabel('Kills', size=14, color="#263238")

ax.set_ylabel('winPlacePerc', size=14, color="#263238")

ax.set_title('Box Plot, Average winPlacePerc of Players VS kills', size=18, color="#263238")

plt.show()
train[train['kills']==72]
data = train[['kills','damageDealt']].copy()

data.loc[data['kills'] > data['kills'].quantile(0.99), 'kills'] = '8+'

x_order = data.groupby('kills').mean().reset_index()['kills']

fig, ax = plt.subplots(figsize=(20,8))

a = sb.boxplot(x='kills', y='damageDealt', data=data, ax=ax, color="#2196F3", order=x_order)

ax.set_xlabel('Kills', size=14, color="#263238")

ax.set_ylabel('damageDealt', size=14, color="#263238")

ax.set_title('[Box Plot] Kills Vs Damage Dealt', size=18, color="#263238")

plt.show()
data = train[['damageDealt','winPlacePerc']].copy()



print("99th percentile of damageDealt is {}".format(data['damageDealt'].quantile(0.99)))



cut = np.linspace(0,780,10)

cut= np.append(cut, 6616)



data['damageDealtGrouping'] = pd.cut(data['damageDealt'],

                                 cut,

                                 labels=["{}-{}".format(a, b) for a, b in zip(cut[:10], cut[1:])],

                                 include_lowest=True

                                )



fig, ax = plt.subplots(figsize=(15,10))

sb.boxplot(x="winPlacePerc", y="damageDealtGrouping", data=data, ax=ax, color="#2196F3")

ax.set_xlabel('winPlacePerc', size=14, color="#263238")

ax.set_ylabel('damageDealt Range Group', size=14, color="#263238")

ax.set_title('Horizontal Box Plot, Win Place Percentile vs Damage Dealt', size=18, color="#263238")

plt.gca().xaxis.grid(True)

plt.show()
data = train[['walkDistance','winPlacePerc']].copy()



print("99th percentile of walkDistance is {}".format(data['walkDistance'].quantile(0.99)))



cut = np.linspace(0,4400,10)

cut= np.append(cut, 26000)

data['walkDistanceGrouping'] = pd.cut(data['walkDistance'],

                                 cut,

                                 labels=["{}-{}".format(a, b) for a, b in zip(cut[:10], cut[1:])],

                                 include_lowest=True

                                )



fig, ax = plt.subplots(figsize=(15,10))

sb.boxplot(x="winPlacePerc", y="walkDistanceGrouping", data=data, ax=ax, color="#2196F3")

ax.set_xlabel('winPlacePerc', size=14, color="#263238")

ax.set_ylabel('walkDistance Range Group', size=14, color="#263238")

ax.set_title('Horizontal Box Plot, Win Place Percentile vs Walk Distance', size=18, color="#263238")

plt.gca().xaxis.grid(True)

plt.show()
data = train[['killPlace','winPlacePerc']].copy()



print("99th percentile of killPlace is {}".format(data['killPlace'].quantile(0.99)))



cut = np.linspace(0,101,10)

data['killPlaceGrouping'] = pd.cut(data['killPlace'],

                                 cut,

                                 labels=["{}-{}".format(a, b) for a, b in zip(cut[:10], cut[1:])],

                                 include_lowest=True

                                )



fig, ax = plt.subplots(figsize=(15,10))

sb.boxplot(x="winPlacePerc", y="killPlaceGrouping", data=data, ax=ax, color="#2196F3")

ax.set_xlabel('winPlacePerc', size=14, color="#263238")

ax.set_ylabel('killPlace Range Group', size=14, color="#263238")

ax.set_title('Horizontal Box Plot, Win Place Percentile vs killPlace', size=18, color="#263238")

plt.gca().xaxis.grid(True)

plt.show()
data = train[['boosts','winPlacePerc']].copy()

data.loc[data['boosts'] >= 10, 'boosts'] = '10+'

order = data.groupby('boosts').mean().reset_index()['boosts']

fig, ax = plt.subplots(figsize=(20,8))

a = sb.boxplot(x='boosts', y='winPlacePerc', data=data, ax=ax, color="#2196F3", order=order)

ax.set_xlabel('boosts', size=14, color="#263238")

ax.set_ylabel('winPlacePerc', size=14, color="#263238")

ax.set_title('Box Plot, Average winPlacePerc of Players VS boosts', size=18, color="#263238")

plt.show()
data = train[['heals','winPlacePerc']].copy()

print(data['heals'].quantile(0.99))

data.loc[data['heals'] >= 12, 'heals'] = '12+'

order = data.groupby('heals').mean().reset_index()['heals']

fig, ax = plt.subplots(figsize=(20,8))

a = sb.boxplot(x='heals', y='winPlacePerc', data=data, ax=ax, color="#2196F3", order=order)

ax.set_xlabel('heals', size=14, color="#263238")

ax.set_ylabel('winPlacePerc', size=14, color="#263238")

ax.set_title('Box Plot, Average winPlacePerc of Players VS heals', size=18, color="#263238")

plt.show()
data = train[['weaponsAcquired','winPlacePerc']].copy()

data.loc[data['weaponsAcquired'] >= 11, 'weaponsAcquired'] = '11+'

order = data.groupby('weaponsAcquired').mean().reset_index()['weaponsAcquired']

fig, ax = plt.subplots(figsize=(20,8))

a = sb.boxplot(x='weaponsAcquired', y='winPlacePerc', data=data, ax=ax, color="#2196F3", order=order)

ax.set_xlabel('weaponsAcquired', size=14, color="#263238")

ax.set_ylabel('winPlacePerc', size=14, color="#263238")

ax.set_title('Box Plot, Average winPlacePerc of Players VS weaponsAcquired', size=18, color="#263238")

plt.show()
sb.set()

cols = ['winPlacePerc', 'revives', 'swimDistance', 'numGroups', 'rankPoints', 'winPoints']

sb.pairplot(train[cols], size = 2.5)

plt.show()
train['matchType'].replace(['squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp'], 'squad',inplace=True)

train['matchType'].replace(['duo-fpp', 'normal-duo-fpp', 'normal-duo', 'crashfpp', 'crashtpp'], 'duo',inplace=True)

train['matchType'].replace(['solo-fpp','normal-solo','normal-solo-fpp'], 'solo',inplace=True)

test['matchType'].replace(['squad-fpp', 'normal-squad-fpp', 'normal-squad', 'flarefpp', 'flaretpp'], 'squad',inplace=True)

test['matchType'].replace(['duo-fpp', 'normal-duo-fpp', 'normal-duo', 'crashfpp', 'crashtpp'], 'duo',inplace=True)

test['matchType'].replace(['solo-fpp','normal-solo','normal-solo-fpp'], 'solo',inplace=True)
train.drop(['Id', 'groupId', 'matchId'], axis=1, inplace = True)

x = train.drop(['winPlacePerc'], axis=1).sample(180000).values

y = train['winPlacePerc'].sample(180000).values
x_linear = LabelEncoder()

x[:, 12] = x_linear.fit_transform(x[:, 12])

hot_enc = OneHotEncoder(categorical_features=[12])

x = hot_enc.fit_transform(x).toarray()

# To avoid Dummy trap

x = x[:, 1:]

# Split the data into train and test set

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = .2, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 1000,

                                  min_samples_split=5,

                                  max_depth = 19,

                                  random_state=101,

                                  n_jobs = -1)

regressor.fit(train_x,train_y)
imp_vals,_ = feature_importance_permutation(predict_method=regressor.predict,

                                          X=test_x,

                                          y=test_y, metric='r2',

                                          num_rounds=1)

feature_importance = pd.DataFrame(imp_vals, index=['solo','squad','assists','boosts','damageDealt',

                                  'DBNOs','headshotKills','heals','killPlace',

                                  'killPoints','kills','killStreaks',

                                  'longestKill','matchDuration','maxPlace',

                                  'numGroups','rankPoints','revives',

                                  'rideDistance','roadKills','swimDistance',

                                  'teamKills','vehicleDestroys','walkDistance',

                                  'weaponAcquired','winPoints'])

feature_importance.plot(kind='bar', figsize=(15,6),

                       title='Features & Their Importance in Win Prediction')
x = train.drop(['headshotKills',

                      'revives','swimDistance','teamKills','winPlacePerc'], axis=1).head(150000).values

y = train['winPlacePerc'].head(150000).values

x_linear = LabelEncoder()

x[:, 11] = x_linear.fit_transform(x[:, 11])

hot_enc = OneHotEncoder(categorical_features=[11])

x = hot_enc.fit_transform(x).toarray()

x_test = test.drop(['Id', 'groupId', 'matchId','headshotKills',

                      'revives','swimDistance','teamKills'], axis=1).values

x_linear_new = LabelEncoder()

x_test[:, 11] = x_linear_new.fit_transform(x_test[:, 11])

hot_enc_new = OneHotEncoder(categorical_features=[11])

x_test = hot_enc_new.fit_transform(x_test).toarray()
regressor = RandomForestRegressor(n_estimators = 1000,

                                  min_samples_split=5,

                                  max_depth = 19,

                                  random_state=101,

                                  n_jobs = -1)



regressor.fit(x,y)

y_test_pred=regressor.predict(x_test)

submissions = test['Id']

submissions=pd.DataFrame(submissions)

submissions['winPlacePerc']=pd.DataFrame(y_test_pred)

submissions.to_csv('submission.csv', index=False)