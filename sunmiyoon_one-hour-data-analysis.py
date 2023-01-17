import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(2)
train.dtypes
train.describe()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for col, a in zip(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'], axes.flatten()):
    a.set_title(col)
    train[train['Survived'] == 1][col].hist(ax=a)
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for col, a in zip(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'], axes.flatten()):
    a.set_title(col)
    train[train['Survived'] == 0][col].hist(ax=a)
def nullCol(df):
    return [col for col in list(df.columns) if (df[col].isnull().sum() != 0)]
print('col which have null values in train set: {}'.format(nullCol(train)))
print('col which have null values in test set: {}'.format(nullCol(test)))
# Average age by Sex with pandas groupby
aveAge = train[['Sex', 'Age']].groupby('Sex').mean()
aveFemaleAge = aveAge.at['female', 'Age']
aveMaleAge = aveAge.at['male', 'Age']
aveAge
aveFare = train[['Pclass', 'Fare']].groupby('Pclass').mean()
aveFirstFare = aveFare.at[1, 'Fare']
aveSecondFare = aveFare.at[2, 'Fare']
aveThirdFare = aveFare.at[3, 'Fare']
aveFare
# Frequency with collections.Counter()
import collections
ports = collections.Counter(train['Embarked'])
mostPopularPort = max(ports, key=ports.get)
ports
# Insert average age by Sex instead of null
train.loc[(train['Age'].isnull()) & (train['Sex'] == 'female'), 'Age'] = aveFemaleAge
train.loc[(train['Age'].isnull()) & (train['Sex'] == 'male'), 'Age'] = aveMaleAge
test.loc[(test['Age'].isnull()) & (test['Sex'] == 'female'), 'Age'] = aveFemaleAge
test.loc[(test['Age'].isnull()) & (test['Sex'] == 'male'), 'Age'] = aveMaleAge

# Insert port which is most people on Titanic came from instead of null
train.loc[(train['Embarked'].isnull()), 'Embarked'] = mostPopularPort

# Insert average fare by ticket class instead of null
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 1), 'Fare'] = aveFirstFare
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 2), 'Fare'] = aveSecondFare
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 3), 'Fare'] = aveThirdFare
# make dummy and erase previous column
def addDummies(df, cols):
    for col in cols:
        dum = pd.get_dummies(df[col])
        dum.columns = [i + col for i in dum.columns]
        df = df.join(dum).drop([col], axis=1)
    return df
train = addDummies(train, ['Sex', 'Embarked'])
test = addDummies(test, ['Sex', 'Embarked'])
# all cabin in training set
train['Cabin'].unique()

# all cabin in test set
# test['Cabin'].unique()
import numpy as np
train['Cabin'] = train['Cabin'].apply(lambda x: 'Z' if x is np.nan else x[0])
test['Cabin'] = test['Cabin'].apply(lambda x: 'Z' if x is np.nan else x[0])
train = addDummies(train, ['Cabin'])
test = addDummies(test, ['Cabin'])
# consider whole family size
train['Family'] = train['SibSp'] + train['Parch']
train['Family'] = train['SibSp'] + train['Parch']
# I use ticket number to find people who board on Titanic with someone else, not alone.
withPeer = train.groupby('Ticket').count()['PassengerId'].reset_index()
withPeer['peer'] = withPeer['PassengerId'] > 1
train = train.merge(withPeer[['Ticket', 'peer']], on='Ticket')

withPeer = test.groupby('Ticket').count()['PassengerId'].reset_index()
withPeer['peer'] = withPeer['PassengerId'] > 1
test = test.merge(withPeer[['Ticket', 'peer']], on='Ticket')
train.columns
indep = [col for col in test.columns if col not in ['PassengerId', 'Survived', 'Name', 'Ticket']]
indep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ranfor = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(ranfor, train[indep], train['Survived'])
print("cross validation score: {}".format(scores))
trees = ranfor.fit(train[indep], train['Survived'])
pred = trees.predict(test[indep])
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })

submission.to_csv('./submission.csv', index=False)