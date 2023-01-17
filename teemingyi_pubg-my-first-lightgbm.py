# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read the data
data = pd.read_csv('../input/train_V2.csv')
data.shape
data = data.dropna()
data.shape
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_sample = data.sample(frac=0.1)
data_sample.head()
data_sample.matchType.unique()
def merge_matchType(x):
    if x in {'normal-squad-fpp', 'crashfpp', 'crashtpp', 'normal-duo-fpp',
       'flarefpp', 'normal-solo-fpp', 'flaretpp', 'normal-duo',
       'normal-squad', 'normal-solo'}:
        return 'others'
    else:
        return x
data_sample['matchType'] = data_sample.matchType.apply(merge_matchType)

data_sample.matchType.unique()
data_sample_dumm = pd.get_dummies(data_sample, columns=['matchType'])
data_sample_dumm.head()
data_sample_dumm = data_sample_dumm.drop('matchType_others', axis=1)
data_sample_dumm.columns
data_sample = data_sample_dumm.loc[:,['Id', 'groupId', 'matchId', 'assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups',
       'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance',
       'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
       'winPoints', 'matchType_duo', 'matchType_duo-fpp',
       'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
       'matchType_squad-fpp', 'winPlacePerc']]
print(data_sample.shape)
data_sample.head()
# split data into X and y
X = data_sample.iloc[:,3:33]
Y = data_sample.iloc[:,33]
X.head()
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
d_train = lgb.Dataset(X_train, label=y_train)

params = {}
params['objective'] = 'regression'
params['metric'] = 'mae'
model = lgb.train(params, d_train)
#Prediction
y_pred=model.predict(X_test)
y_pred
pd.Series(y_pred).plot(kind='hist',bins=10)
pd.Series(y_test).plot(kind='hist', bins=10)
# evaluate predictions
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE: {}".format(mae))
lgb.plot_importance(model)
lgb.create_tree_digraph(model)
data_pred = pd.concat([X_test, y_test], axis=1)
data_pred = pd.concat([data_pred.reset_index(), pd.Series(y_pred)],axis=1)
data_pred.head()
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
data_pred['pred_winPlacePerc'] = data_pred.iloc[:,-1]
def adjust_pred(x):
    space = 1/(x.maxPlace-1)
    return round(x.pred_winPlacePerc / space) * space

data_pred['adj_winPlacePerc'] = adjust_pred(data_pred)
mae = mean_absolute_error(data_pred.winPlacePerc, data_pred.adj_winPlacePerc)
print("MAE: {}".format(mae))