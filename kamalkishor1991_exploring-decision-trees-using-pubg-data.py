# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head()
train.shape, test.shape
# train = train[:500000]
test.head()
train.columns
target_col = 'winPlacePerc'
train.drop(2744604, inplace=True, errors='ignore')
train.dtypes
train['Id'].describe()
train['groupId'].describe()
train['matchId'].describe()

train['matchType'].describe()
train['matchType'].unique()
pd.get_dummies(train['matchType']).head() # getting one hot encoded dataframe
# Get one hot encoding of columns matchType
one_hot_train = pd.get_dummies(train['matchType'])
# Drop column B as it is now encoded
train = train.drop('matchType', axis = 1)
# Join the encoded df
train = train.join(one_hot_train)
train.head()
# Get one hot encoding of columns matchType
one_hot_test = pd.get_dummies(test['matchType'])
# Drop column B as it is now encoded
test = test.drop('matchType', axis = 1)
# Join the encoded df
test = test.join(one_hot_test)
test.head()
print("Adding Features")
def add_feature(df):
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    print("Removing Na's From DF")
    df.fillna(0, inplace=True)
#head = train.head()
# add_feature(train)
# add_feature(test)
#head
train.shape, test.shape
from sklearn.model_selection import train_test_split
train, val = train_test_split(train, test_size=0.2)
trainIds = train['Id']
testIds = test['Id']
#??train.drop
x_train = train.drop(['Id', 'groupId', 'matchId', target_col], axis=1)
x_train.head(1)
y_train = train[target_col]
y_train.head()
x_val = val.drop(['Id', 'groupId', 'matchId', target_col], axis=1)
x_val.head(1)
y_val = val[target_col]
y_val.head()
x_test = test.drop(['Id', 'groupId', 'matchId'], axis=1)
x_test.head(1)
x_train.shape, x_test.shape
from sklearn.tree import DecisionTreeRegressor
#%timeit 
dtree=DecisionTreeRegressor(max_depth=3)
dtree.fit(x_train, y_train)
dtree.score(x_train, y_train)
dtree.score(x_val, y_val)
from IPython.display import SVG
from graphviz import Source
from sklearn import tree
graph = Source( tree.export_graphviz(dtree, out_file=None, feature_names=x_train.columns))
SVG(graph.pipe(format='svg'))
p_val = dtree.predict(x_val)
def mae(p, t):
    return np.sum(np.abs(p - t)) / len(p)
mae(p_val, y_val)
dtree=DecisionTreeRegressor(max_depth=15)
dtree.fit(x_train, y_train)
def print_score(mm):
    print("train r^2 " + str(mm.score(x_train, y_train)))
    print("validation r^2 " + str(mm.score(x_val, y_val)))
    p_val = mm.predict(x_val)
    p_train = mm.predict(x_train)
    print("mean absolute error(Train): " + str(mae(p_train, y_train)))
    print("mean absolute error(Validation): " + str(mae(p_val, y_val)))
print_score(dtree)
p_test = dtree.predict(x_test)
p_test
submission = pd.DataFrame()
submission['Id'] = testIds
submission[target_col] = p_test
submission.head()
submission.to_csv("submission.csv", index=False)
def train_and_get_score(max_depth):
    dtree=DecisionTreeRegressor(max_depth=max_depth)
    dtree.fit(x_train[:10000], y_train[:10000])
    p_val = dtree.predict(x_val[:1000])
    p_train = dtree.predict(x_train[:10000])
    return mae(p_train, y_train[:10000]), mae(p_val, y_val[:1000])
train_and_get_score(5)
train_score = []
valid_score = []
x_axis = []
for max_depth in range(1, 40, 5):
    ts, vs = train_and_get_score(max_depth)
    train_score.append(ts)
    valid_score.append(vs)
    x_axis.append(max_depth)
import matplotlib.pyplot as plt
plt.plot(x_axis, train_score,)
plt.plot(x_axis, valid_score)
plt.ylabel('Score')
plt.xlabel('max_depth')
plt.show()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, max_depth=15, max_features='sqrt', n_estimators=100)
%time model.fit(x_train, y_train)
print_score(model)
model.estimators_[0]
print_score(model.estimators_[0])
#for dtree_model in model.estimators_:
#    print_score(dtree_model)
#    print("------------------------------------")
p_test = model.predict(x_test)
submission = pd.DataFrame()
submission['Id'] = testIds
submission[target_col] = p_test
submission.to_csv("submission_random_forest_30.csv", index=False)
feature_imp = pd.DataFrame(model.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_imp
feature_imp['importance'].nlargest(10).plot(kind='barh')
#import xgboost
# xgb = xgboost.XGBRegressor(n_jobs=-1, n_estimators=100, learning_rate=0.2, gamma=0, subsample=0.75,
                           #colsample_bytree=1, max_depth=3)
# %time xgb.fit(x_train, y_train)
#print_score(xgb)
from lightgbm import LGBMRegressor
params = {
    'n_estimators': 100,
    'learning_rate': 0.3, 
    'num_leaves': 30,
    'objective': 'regression_l2', 
    'metric': 'mae',
    'verbose': -1,
}

model = LGBMRegressor(**params)
model.fit(
   x_train, y_train,
    eval_metric='mae',
    verbose=20,
)
print_score(model)
p_test = model.predict(x_test)
submission = pd.DataFrame()
submission['Id'] = testIds
submission[target_col] = p_test
submission.to_csv("submission_lightgdm.csv", index=False)
feature_imp = pd.DataFrame(model.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_imp['importance'].nlargest(10).plot(kind='barh')