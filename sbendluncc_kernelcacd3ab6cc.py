# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_df.head()
train.head()
test.head()
train.describe()
stringExample = "DeGrace, Mister Stephen Bendl"
stringExample.split(".")
train.Name
train['last_name'] = train.Name.str.split(',').apply(lambda x: x[0])
train['title'] = train.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0])

test['title'] = test.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0])
train['first_name'] = train.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[1]).str.split().apply(lambda x: x[0])
train['middle_name'] = train.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[1]).str.split().apply(lambda x: x[1] if len(x) > 1 else '')
train.Name
train['maiden_name'] = train.Name.str.extract('\((.*)\)').fillna('')

train.maiden_name
train
def split_train_score(X, y):

    X = X.values

    y = y.values

    kf = StratifiedKFold(2)

    dummy_scores = []

    scores = []

    for train_index, test_index in kf.split(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived']):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4)

#         clf = LogisticRegressionCV(max_iter=10000)

        dum = DummyClassifier()

        clf.fit(X_train, y_train)

        dum.fit(X_train, y_train)

        scores.append(clf.score(X_test, y_test))



        dummy_scores.append(dum.score(X_test, y_test))

    

    print('Classifier: ', sum(scores)/len(scores))

    print('Dummy: ', sum(dummy_scores)/len(dummy_scores))
train_processed = train.copy()

test_processed = test.copy()
train_numeric = train_processed[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']]

test_numeric =  test_processed[['Age', 'SibSp', 'Parch', 'Fare']]
train_processed = train_numeric.join(pd.get_dummies(train[['Pclass', 'Sex', 'Embarked']].astype('str')))

test_processed = test_numeric.join(pd.get_dummies(test[['Pclass', 'Sex', 'Embarked']].astype('str')))
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.dummy import DummyClassifier

from sklearn.metrics import r2_score
train_processed = train_numeric.join(pd.get_dummies(train[['Pclass', 'Sex', 'title', 'Embarked']].astype('str')))

test_processed = test_numeric.join(pd.get_dummies(test[['Pclass', 'Sex', 'title', 'Embarked']].astype('str')))
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
train_processed.fillna(train_processed.mean(), inplace=True)

test_processed.fillna(test_processed.mean(), inplace=True)
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
train['floor'] = train.Cabin.apply(lambda x: x[0] if type(x) == str else 'unk')

test['floor'] = test.Cabin.apply(lambda x: x[0] if type(x) == str else 'unk')
train_processed = train_numeric.join(pd.get_dummies(train[['Pclass', 'Sex', 'title', 'Embarked', 'floor']].astype('str')))

test_processed = test_numeric.join(pd.get_dummies(test[['Pclass', 'Sex', 'title', 'Embarked', 'floor']].astype('str')))
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
train[train.Ticket.str.split().apply(lambda x: len(x) > 1)].Ticket.str.split().apply(lambda x: x[0]).unique()
train['ticket_prefix'] = train.Ticket.str.split().apply(lambda x: x[0] if len(x) > 1 else 'N/A')

test['ticket_prefix'] = test.Ticket.str.split().apply(lambda x: x[0] if len(x) > 1 else 'N/A')
train_processed = train_numeric.join(pd.get_dummies(train[['Pclass', 'Sex', 'title', 'Embarked', 'floor', 'ticket_prefix']].astype('str')))

test_processed = test_numeric.join(pd.get_dummies(test[['Pclass', 'Sex', 'title', 'Embarked', 'floor', 'ticket_prefix']].astype('str')))
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
def parse_cabin(cabin):

    try:

        if type(cabin) == str:

            cabin_side = int(cabin[-1])

            cabin_side = cabin_side % 2

            cabin_side = not bool(cabin_side)

        else:

            cabin_side = 'unk'

    except Exception as e:

        print(cabin, e)

        cabin_side = 'unk'

    

    

    return cabin_side if cabin_side == 'unk' else 'starbord' if cabin_side == True else 'port'
parse_cabin('c124')
train['cabin_side'] = train.Cabin.apply(parse_cabin)

test['cabin_side'] = test.Cabin.apply(parse_cabin)
train_processed = train_numeric.join(pd.get_dummies(train[['Pclass', 'Sex', 'title', 'Embarked', 'floor', 'ticket_prefix', 'cabin_side']].astype('str'))).drop(['title_ Capt','title_ Don','title_ Jonkheer','title_ Lady','title_ Major','title_ Mlle','title_ Mme','title_ Sir','title_ the Countess','Embarked_nan','floor_T','ticket_prefix_A/4.','ticket_prefix_A/S','ticket_prefix_A4.','ticket_prefix_C.A./SOTON','ticket_prefix_Fa','ticket_prefix_P/PP','ticket_prefix_S.C./A.4.','ticket_prefix_S.O.P.','ticket_prefix_S.P.','ticket_prefix_S.W./PP','ticket_prefix_SCO/W','ticket_prefix_SO/C','ticket_prefix_SW/PP','ticket_prefix_W/C','ticket_prefix_WE/P'], axis=1)

test_processed = test_numeric.join(pd.get_dummies(test[['Pclass', 'Sex', 'title', 'Embarked', 'floor', 'ticket_prefix', 'cabin_side']].astype('str'))).drop(['title_ Dona','ticket_prefix_A.','ticket_prefix_AQ/3.','ticket_prefix_AQ/4','ticket_prefix_LP','ticket_prefix_SC/A.3','ticket_prefix_SC/A4','ticket_prefix_STON/OQ.'], axis=1)
split_train_score(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4)
clf.fit(train_processed.drop('Survived', axis=1).fillna(0), train_processed['Survived'])
test['Survived'] = clf.predict(test_processed.fillna(0))
submission = test[['PassengerId', 'Survived']]
submission
submission.to_csv('submission.csv', index=False)