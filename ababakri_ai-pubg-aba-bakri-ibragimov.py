%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import misc

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KDTree, BallTree, KNeighborsClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
train = pd.read_csv('../input/train_V2.csv')

train.shape
train.head()
train = train[['assists', 'damageDealt', 'heals', 'longestKill', 'matchDuration', 'revives','walkDistance','winPlacePerc']]

train.isnull().any()
train['winPlacePerc'].interpolate(inplace = True)

train.isnull().any()
# train['matchType'].replace('solo', 0, inplace = True)

# train['matchType'].replace('solo-fpp', 0, inplace = True)

# train['matchType'].replace('normal-solo', 0, inplace = True)

# train['matchType'].replace('normal-solo-fpp', 0, inplace = True)

# train['matchType'].replace('duo', 1, inplace = True)

# train['matchType'].replace('duo-fpp', 1, inplace = True)

# train['matchType'].replace('normal-duo', 1, inplace = True)

# train['matchType'].replace('normal-duo-fpp', 1, inplace = True)

# train['matchType'].replace('squad', 2, inplace = True)

# train['matchType'].replace('squad-fpp', 2, inplace = True)

# train['matchType'].replace('normal-squad-fpp', 2, inplace = True)

# train['matchType'].replace('normal-squad', 2, inplace = True)

# train['matchType'].replace('crashfpp', 3, inplace = True)

# train['matchType'].replace('crashtpp', 3, inplace = True)

# train['matchType'].replace('flaretpp', 4, inplace = True)

# train['matchType'].replace('flarefpp', 4, inplace = True)

train.head()
damageDealt = train['damageDealt'].values.reshape(-1,1)

min_max_scaler = preprocessing.MinMaxScaler()

damageDealt_scaled = min_max_scaler.fit_transform(damageDealt)

train['damageDealt'] = pd.DataFrame(damageDealt_scaled)



longestKill = train['longestKill'].values.reshape(-1,1)

min_max_scaler2 = preprocessing.MinMaxScaler()

longestKill_scaled = min_max_scaler2.fit_transform(longestKill)

train['longestKill'] = pd.DataFrame(longestKill_scaled)



matchDuration = train['matchDuration'].values.reshape(-1,1)

min_max_scaler3 = preprocessing.MinMaxScaler()

matchDuration_scaled = min_max_scaler3.fit_transform(matchDuration)

train['matchDuration'] = pd.DataFrame(matchDuration_scaled)



walkDistance = train['walkDistance'].values.reshape(-1,1)

min_max_scaler3 = preprocessing.MinMaxScaler()

walkDistance_scaled = min_max_scaler3.fit_transform(walkDistance)

train['walkDistance'] = pd.DataFrame(walkDistance_scaled)



train.head()

trainX, testX, trainY, testY = train_test_split(train[['assists', 'damageDealt', 'heals', 'longestKill','matchDuration','revives','walkDistance']], train['winPlacePerc'], test_size = 0.3)
lr = LinearRegression()

lr.fit(trainX, trainY)

predictions = lr.predict(trainX)

print(predictions)

lr.score(testX,testY)
test = pd.read_csv('../input/test_V2.csv')

test.shape
idd=test['Id']
idd
trainX = train[['assists', 'damageDealt', 'heals', 'longestKill', 'matchDuration', 'revives','walkDistance']]

trainY = train['winPlacePerc']

testX = test[['assists', 'damageDealt', 'heals', 'longestKill', 'matchDuration', 'revives','walkDistance']]
result = pd.DataFrame(idd)

lr = LinearRegression()

lr.fit(trainX, trainY)

predictions = lr.predict(testX)

result['winPlacePerc'] = predictions

result.to_csv('./sampleSubmission3.csv', index = False)