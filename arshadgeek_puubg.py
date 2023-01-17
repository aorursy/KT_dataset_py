import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv("../input/train_V2.csv")

test  = pd.read_csv("../input/test_V2.csv")

print(train.shape)

print(test.shape)
var = 'assists'

data = pd.concat([train['winPlacePerc'], train[var]], axis=1)

data.plot.scatter(x=var, y='winPlacePerc', ylim=(0,1));
var = 'boosts'

data = pd.concat([train['winPlacePerc'], train[var]], axis=1)

data.plot.scatter(x=var, y='winPlacePerc', ylim=(0,1));
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(29)
#bivariate analysis saleprice/grlivarea

var = 'walkDistance'

data = pd.concat([train['winPlacePerc'], train[var]], axis=1)

data.plot.scatter(x=var, y='winPlacePerc', ylim=(0,1));
print(train.head(5))
X_train = train.drop(['Id','groupId','matchId','headshotKills','killPlace', 'killPoints','killStreaks',

                      'matchDuration','matchType','maxPlace',

                      'numGroups','rankPoints','rideDistance','roadKills',

                      'swimDistance','teamKills','vehicleDestroys','winPoints','winPlacePerc'],

                     axis = 1)
Y_train = train['winPlacePerc']
train['winPlacePerc'].fillna((train['winPlacePerc'].mean()), inplace=True)
# with sklearn

from sklearn import linear_model



regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)

test = pd.read_csv("../input/test_V2.csv")
X_test = test.drop(['Id','groupId','matchId','headshotKills','killPlace', 'killPoints','killStreaks',

                      'matchDuration','matchType','maxPlace',

                      'numGroups','rankPoints','rideDistance','roadKills',

                      'swimDistance','teamKills','vehicleDestroys','winPoints'],

                     axis = 1)
predictions = regr.predict(X_test)

print(predictions)
print(predictions.shape)
my_submission = pd.DataFrame({'Id': test.Id, 'winPlacePerc': predictions})



my_submission.to_csv('submission.csv', index=False)