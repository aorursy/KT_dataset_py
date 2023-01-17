import numpy as np

import pandas as pd

import sklearn
train = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
train.head()
test.head()
train.isnull().any()
test.isnull().any()
X_train = train.drop('id',axis=1)

X_test = test.drop('id',axis=1)
y_train = X_train['class']
X_train.head()
X_train = X_train.drop('class',axis=1)
X_train.head()


y_train.head()
X_test.head()
from sklearn.ensemble import ExtraTreesClassifier



clf = ExtraTreesClassifier(n_estimators = 1000,min_samples_split=2, min_samples_leaf=2, max_features='sqrt',

                           max_depth=30, bootstrap=False)





clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
res = pd.DataFrame(y_pred, columns=['class'])

res = pd.concat([test['id'],res],axis=1)
res.head()
res.to_csv('submission.csv',index=False)
