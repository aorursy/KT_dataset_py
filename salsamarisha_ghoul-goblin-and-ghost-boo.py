import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
for column in train.columns:

    print(column, train[column].isnull().sum())
for column in test.columns:

    print(column, test[column].isnull().sum())
train.info()
test.info()
train_y = train['type']

train_x = pd.DataFrame(train.drop('type', axis=1))
train_test=pd.concat([train_x,test])
train_test.head()
to_dummies_col = ['color']
train_test_dum = pd.get_dummies(train_test, columns=to_dummies_col)
train_test_dum.head()
train_x = train_test_dum[:len(train)]

test_x = train_test_dum[len(train):]
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



logreg = LogisticRegression()



import numpy as np

clf = logreg

clf.fit(train_x, np.ravel(train_y))

score = cross_val_score(clf, train_x, np.ravel(train_y), cv=5, scoring = 'accuracy')
np.mean(score)

predict=clf.predict(test_x)
submission = pd.DataFrame()

submission['id']=test['id']

submission['type']=predict
submission.info()
submission.head()
submission.to_csv('submission.csv', index= False)