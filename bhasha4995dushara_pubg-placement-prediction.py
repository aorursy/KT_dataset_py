import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import cufflinks as cf
import plotly

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
print(train.shape)
train.head()
train.info()
train['winPlacePerc'].describe()
test = pd.read_csv('../input/test_V2.csv')
print(test.shape)
test.head()
train.isnull().sum().sum()
test.isnull().sum().sum()
train.winPlacePerc.fillna(1,inplace=True)
train.loc[train['winPlacePerc'].isnull()]

train["distance"] = train["rideDistance"]+train["walkDistance"]+train["swimDistance"]
train["skill"] = train["headshotKills"]+train["roadKills"]
train.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(train.shape)
train.head()
test["distance"] = test["rideDistance"]+test["walkDistance"]+test["swimDistance"]
test["skill"] = test["headshotKills"]+test["roadKills"]
test.drop(['rideDistance','walkDistance','swimDistance','headshotKills','roadKills'],inplace=True,axis=1)
print(test.shape)
test.head()
corrmat = train.corr() 
cols = corrmat.nlargest(26, 'winPlacePerc').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(train[cols].values.T)

# correlation 
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(15, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()
predictors = [ "kills",
                "maxPlace",
                "numGroups",
                "distance",
                "boosts",
                "heals",
                "revives",
                "killStreaks",
                "weaponsAcquired",
                "winPoints",
                "skill",
                "assists",
                "damageDealt",
                "DBNOs",
                "killPlace",
                "killPoints",
                "vehicleDestroys",
                "longestKill"
               ]
print(len(predictors))
X_train = train[predictors]
X_train.head()
y_train = train['winPlacePerc']
y_train.head()
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
lgb_reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 20, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.8,metric='mse')
lgb_reg.fit(X_train, y_train)
X_test = test[predictors]
X_test.head()
y_pred = lgb_reg.predict(X_test)
y_pred
len(y_pred[y_pred > 1])
y_pred[y_pred > 1] = 1
len(y_pred[y_pred > 1])
ss = ShuffleSplit(n_splits=10)
scores = cross_val_score(lgb_reg, X_train, y_train, cv=ss)
print(scores)
accuracy = scores.mean()
print(accuracy)
lgb.plot_importance(lgb_reg, max_num_features=20, figsize=(12, 10),xlabel='Features Importance',ylabel='Features')
plt.title('Feature importance')
test_id = test["Id"]
submit = pd.DataFrame({'Id': test_id, "winPlacePerc": y_pred} , columns=['Id', 'winPlacePerc'])
print(submit.head())

submit.to_csv("submission.csv", index = False)