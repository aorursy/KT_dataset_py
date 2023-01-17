import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from plotnine import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import export_graphviz
pd.set_option('float_format', '{:f}'.format)
%matplotlib inline
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.describe(include='all')
train['totalPlayers'] = train['matchId'].map(train.groupby('matchId')['Id'].count())
#train['totalTeams'] = train['matchId'].map(train.groupby('matchId')['groupId'].nunique())
train['playersInTeam'] = train.groupby(['matchId', 'groupId'])['Id'].transform('count')
train['maxPlayersInTeam'] = train.groupby('matchId')['playersInTeam'].transform('max')
train['totalKills'] = train['matchId'].map(train.groupby('matchId')['kills'].sum())
X_train = train.drop(['Id', 'winPlacePerc'], axis=1)
Y_train = train['winPlacePerc']
X_train, X_cv, y_train, y_cv = train_test_split( X_train, Y_train, test_size = 0.3, random_state = 100)
y_train.head()
dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(X_train, y_train)
y_cv_pred = dtr.predict(X_cv)
np.mean(np.abs(y_cv_pred - y_cv))
cols = list(X_train.columns.values)
tree.export_graphviz(dtr, out_file='tunedtreewithdummies.dot',feature_names  = cols) 
'''
# Decision tree tuning
for crtr in ['gini','entropy']:
    for md in [3,4,5]:
        for spltr in ['best','random']:
            for mss in [6,10,16,26,42]:
                for msl in [6,10,16,26,42]:
                    dts = DecisionTreeRegressor(criterion=crtr, max_depth=md,
                                max_features=None, max_leaf_nodes=None, min_samples_leaf=msl,
                                min_samples_split=mss, min_weight_fraction_leaf=0.0,
                                presort=False, random_state=100, splitter=spltr)
                    dts.fit(X_train, y_train)
                    y_pred = dts.predict(X_cv)
                    sip=score_in_percent(y_pred,y_cv)
                    print("score for {} criterion, {} max_depth, {} splitter, {} min_samples_split, {} min_samples_leaf is {}".format(crtr,md,spltr,mss,msl,sip))
'''                    
