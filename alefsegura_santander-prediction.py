import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('Shapes:')

print(train.shape)

print(test.shape)



print('\nIDs únicos:')

print(train.ID.nunique())

print(test.ID.nunique())



print('\nColunas com nulos:')

print((train.isnull().sum()!=0).sum())
train.TARGET.value_counts()
# Features de variância nula

to_drop = train.columns[train.std()==0]

train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)



print('dropped: ' + str(len(to_drop)))

print(train.shape)

print(test.shape)
# colunas duplicadas

to_drop = []

n_cols = len(train.drop(['TARGET'],axis=1).columns)

for i in range(n_cols):

    v1 = train.iloc[i].values

    for j in range(i+1,n_cols):

        v2 = train.iloc[j].values

        if np.all(v1 == v2):

            to_drop.append(train.columns[j])



train.drop(to_drop, axis=1, inplace=True)

test.drop(to_drop, axis=1, inplace=True)



print('dropped: ' + str(len(to_drop)))

print(train.shape)

print(test.shape)
plt.figure(figsize=(11,9))

sns.heatmap(train.corr(),cmap="YlGnBu");
X = train.drop(['TARGET','ID'],axis=1)

y = train['TARGET']



X_train, X_test, y_train, y_test = train_test_split(X,y)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



X = train.drop(['TARGET','ID'],axis=1)

y = train.TARGET

X_train, X_test, y_train, y_test = train_test_split(X,y)
clf = xgb.XGBClassifier(max_depth = 5, n_estimators=1000, learning_rate=0.2, nthread=3, subsample=1.0,

                        colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.03, eed=1301)



clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])
submission = test[['ID']]

submission['TARGET'] = clf.predict_proba(test.drop(['ID'],axis=1), ntree_limit=clf.best_iteration)[:,1]
submission.to_csv('submission.csv', index=False)

submission.head()