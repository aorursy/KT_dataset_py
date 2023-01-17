import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)

train.Cabin.fillna('NO_VALUE', inplace=True)
train.Embarked.fillna('NO_VALUE', inplace=True)
test.Cabin.fillna('NO_VALUE', inplace=True)
test.Embarked.fillna('NO_VALUE', inplace=True)

#train.drop(train.index[train.Age.isnull()], inplace=True)
train.Age.fillna(round(train.Age.mean()), inplace=True)
test.Age.fillna(round(test.Age.mean()),inplace=True)

test.Fare.fillna(round(test.Fare.mean(), 2), inplace=True)
train['sex+class'] = train.Sex + train.Pclass.astype(str)
train['sex+age'] = train.Sex + train.Age.astype(str)
train['sex+sib'] = train.Sex + train.SibSp.astype(str)
train['sex+par'] = train.Sex + train.Parch.astype(str)

test['sex+class'] = test.Sex + test.Pclass.astype(str)
test['sex+age'] = test.Sex + test.Age.astype(str)
test['sex+sib'] = test.Sex + test.SibSp.astype(str)
test['sex+par'] = test.Sex + test.Parch.astype(str)
y = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

passenger_ids = test.PassengerId.copy()
test.drop('PassengerId', axis=1, inplace=True)
vectorizer = DictVectorizer(sparse=False)
vectorizer.fit(train.to_dict(orient='records'))

X_train = vectorizer.transform(train.to_dict(orient='records'))
X_test = vectorizer.transform(test.to_dict(orient='records'))
round(cross_val_score(LogisticRegression(), X_train, y).mean(), 4)
def save_predictions(ids, predictions):
    with open('submission.csv', 'w') as f:
        f.write('PassengerId,Survived\n')
        for p in zip(ids, predictions):
            f.write('{:d},{:d}\n'.format(p[0], p[1]))
estimator = LogisticRegression()
estimator.fit(X_train, y)
predictions = estimator.predict(X_test)
save_predictions(passenger_ids, predictions)
