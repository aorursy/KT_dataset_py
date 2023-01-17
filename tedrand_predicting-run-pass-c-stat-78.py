import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

from sklearn import grid_search

from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('../input/nflplaybyplay2015.csv',low_memory=False)

df.columns
"""

Boiler-Plate/Feature-Engineering to get frame into a testable format

"""



# Only use downs 1-3 since 4th is too unpredictable

used_downs = [1,2,3] # Downs that are being used in predictions

df = df[df['down'].isin(used_downs)]



# Don't include kicks, kneels, spikes, etc.

valid_plays = ['Pass', 'Run', 'Sack']

df = df[df['PlayType'].isin(valid_plays)]



# create a column that has 1 for pass/sack, 0 for run

pass_plays = ['Pass', 'Sack']

df['is_pass'] = df['PlayType'].isin(pass_plays).astype('int')



# select your features and classifier from full data frame

df = df[['down','yrdline100','ScoreDiff', 'PosTeamScore', 'DefTeamScore',

         'ydstogo','TimeSecs','ydsnet','is_pass','Drive']]



# train/test split on data

X, test = train_test_split(df, test_size = 0.2)



# pop the classifier off the sets.

y = X.pop('is_pass')

test_y = test.pop('is_pass')
parameters = {

    # 'n_estimators':[1, 5, 10, 30, 50, 100, 200],

    # 'min_samples_leaf':[10, 12, 14, 16, 18, 20],

    # 'max_features':[.2,.5,.8, 1.0]

             }

clf = RandomForestRegressor(n_jobs = -1, oob_score=True,

    n_estimators=100, min_samples_leaf=12, max_features=.8

)
# clf = grid_search.GridSearchCV(rf, parameters)

clf.fit(X,y)
clf.score(test, test_y)
y_oob = clf.oob_prediction_

print('c-stat:', roc_auc_score(y, y_oob))
x = range(X.shape[1])

sns.barplot(x = clf.feature_importances_, y = X.columns)

sns.despine(left=True, bottom=True)