import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
train_matches = pd.read_csv('../input/sescds-dota/train.csv')

test_matches = pd.read_csv('../input/sescds-dota/test.csv')

gold = pd.read_csv('../input/sescds-dota/gold.csv')
gold = gold[gold.times == 600].reset_index(drop=True)
gold.head()
gold['radiant_gold'] = gold[['player_0', 'player_1', 'player_2', 'player_3', 'player_4']].sum(axis=1)

gold['dire_gold'] = gold[['player_5', 'player_6', 'player_7', 'player_8', 'player_9']].sum(axis=1)
gold.head()
gold['diff_gold'] = gold['radiant_gold'] - gold['dire_gold']

gold['ratio_gold'] = gold['radiant_gold'] / gold['dire_gold']
train = pd.merge(train_matches[['mid']], gold, on='mid', how='left').drop(['mid', 'times'], 1)

test = pd.merge(test_matches[['mid']], gold, on='mid', how='left').drop(['mid', 'times'], 1)
x_train = train.values

x_test = test.values

y_train = train_matches.radiant_won.values
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1234)

np.mean(cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc'))
clf.fit(x_train, y_train)
test_matches['radiant_won'] = clf.predict_proba(x_test)[:, 1]
test_matches.to_csv('submission.csv', index=False)