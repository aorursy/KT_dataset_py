import numpy as np

import pandas as pd
submission = pd.read_csv('../input/predict-the-diabetes/sample_submission.csv')
submission.head()
test = pd.read_csv('../input/predict-the-diabetes/test.csv')
test.head()
len(test)
train = pd.read_csv('../input/predict-the-diabetes/train.csv')
train.head()
len(train)
from sklearn.preprocessing import scale
X_train = scale(train.drop(['Outcome', 'Id'], axis=1))

X_test = scale(test.drop(['Id'], axis=1))

y = train['Outcome']
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier, Pool
clf = CatBoostClassifier(iterations=500,

                           depth=11,

                           learning_rate=1,

                           loss_function='Logloss', verbose=False)



scores = cross_val_score(clf, X_train, y, cv=5)

print(scores)
clf = CatBoostClassifier(iterations=500,

                           depth=8,

                           learning_rate=1,

                           loss_function='Logloss', verbose=False)



clf.fit(X_train, y)
clf.score(X_train, y)
submission['Outcome'] = clf.predict(X_test)
submission
submission.to_csv('submission.csv', index=False)