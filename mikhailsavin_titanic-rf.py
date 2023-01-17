import numpy as np
import pandas as pd 
import os as os
import catboost

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

#preprocessing
drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Cabin']

train = train.drop(drop, axis=1)
test = test.drop(drop, axis=1)

train.columns = map(str.lower, train.columns)
test.columns = map(str.lower, test.columns)

test = pd.get_dummies(test, drop_first=True)
train = pd.get_dummies(train, drop_first=True)
train = train[~train['age'].isna()]

train, valid = train_test_split(train, test_size=0.05, random_state=123)

train_t = train['survived']
train_f = train.drop(['survived', 'passengerid'], axis=1)
valid_t = valid['survived']
valid_f = valid.drop(['survived', 'passengerid'], axis=1)

test_f = test.drop('passengerid', axis=1)
train.corr()
#check best params
scores = []

for tree in range(10, 90, 10):
    for deep in range(2, 15, 1):
        for leaf in range (2, 5):
            for split in range (2, 5):
                model = RandomForestClassifier(n_estimators=tree, 
                                               max_depth=deep,
                                               min_samples_leaf=leaf,
                                               min_samples_split=split,
                                               random_state=123)
                model.fit(train_f, train_t)
                predict = model.predict(valid_f)
                accuracy = accuracy_score(valid_t, predict)

                if len(scores) == 0:
                    scores.append([tree, deep, leaf, split, accuracy])
                else:
                    if scores[-1][4] < accuracy:
                        scores.append([tree, deep, leaf, split, accuracy])
                    else:
                        pass

print('Best score model:')
print('Accuracy: {:.5f}'.format(scores[-1][4]))
print('Params: ', scores[-1][0], scores[-1][1], scores[-1][2], scores[-1][3])
model = RandomForestClassifier(n_estimators=50,
                               max_depth=7,
                               min_samples_leaf=2,
                               min_samples_split=2,
                               random_state=123)
model.fit(train_f, train_t)
predictions = model.predict(valid_f)
score = accuracy_score(valid_t, predictions)
score
#get test dataset result
test_f = test_f.fillna(test['age'].mean())

predictions = model.predict(test_f)
predictions
test['survival'] = pd.Series(predictions)
result = test[['passengerid', 'survival']]
result.columns =  ['PassengerId', 'Survived']
result = result.reset_index(drop=True)

result.to_csv('titanic.csv',index=False)
