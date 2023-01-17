import sys

import numpy as np # linear algebra

import pandas as pd

# Random Forests are good here because very small amount of data

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split # split data

import seaborn as sns; # plotting

%matplotlib inline
df = pd.read_csv("../input/nflplaybyplay2015.csv", low_memory=False)

df.columns.values
mn_o = df[df.posteam == 'MIN'] 
mn_o.PlayType.unique()
valid_plays = ['Pass', 'Run', 'Sack']

mn_ov = mn_o[mn_o.PlayType.isin(valid_plays)] # mn_ov -> offensive plays considered in model
mn_ovn = mn_ov[mn_ov.down.isin([1,2,3])]

mn_ovn = mn_ovn[mn_ovn.TimeSecs>120] # Last two minutes too situational
mn_ovn.describe()
mn_gb = mn_ovn[mn_ovn.DefensiveTeam == 'GB'] 
len(mn_gb)
# create a column that has 1 for pass/sack, 0 for run

pass_plays = ['Pass', 'Sack']

mn_gb['is_pass'] = mn_gb['PlayType'].isin(pass_plays).astype('int')

mn_gb_pred = mn_gb[['down','yrdline100','ScoreDiff', 'PosTeamScore', 'DefTeamScore',

             'ydstogo','TimeSecs','ydsnet','Drive', 'yrdln','is_pass']]



# train/test split on data

X, test = train_test_split(mn_gb_pred, test_size = 0.2)

# pop the classifier off the sets.

y = X.pop('is_pass')

test_y = test.pop('is_pass')
# raise number of n_estimators so that it generates more data via bootstrapping

clf = RandomForestClassifier(n_estimators=100000)
clf.fit(X,y)
clf.score(test,test_y)
sns.barplot(x = clf.feature_importances_, y = X.columns)

sns.despine(left=True, bottom=True)