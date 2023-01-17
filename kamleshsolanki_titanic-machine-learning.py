import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt
os.listdir('../input/titanic/')
train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')
train.head(5)
test.head()
print('Train dataset')

print(train.info())

print('*' * 40)

print('Test dataset')

print(test.info())
def percent_null_value(df):

    total = df.isnull().sum()

    percent = df.isnull().sum() / len(df)

    df = pd.DataFrame([total, percent]).T

    df.columns = ['Total', 'Percent']

    return df
def percent_unique_value(df, col):

    total = df[col].value_counts()

    percent = df[col].value_counts() / len(df)

    df = pd.DataFrame([total, percent]).T

    df.columns = ['Total', 'Percent']

    return df
percent_unique_value(train, 'Survived')
df = percent_null_value(train)

df.head(len(df))
df = percent_null_value(test)

df.head(len(df))
train[train.Embarked.isnull()]
train[(train.Pclass == 1) & (train.Sex == 'female')]['Embarked'].value_counts()
train.Embarked.fillna('S', inplace = True)
test[test.Fare.isnull()]
missing_value = test[(test.Pclass == 3) & (test.Sex == 'male')]['Fare'].mean()

test.Fare.fillna(missing_value, inplace = True)
data = [train, test]

for dataset in data:

    dataset.Cabin.fillna('N', inplace = True)

    dataset['Cabin'] = dataset['Cabin'].apply(lambda x : x[0])
survived = train.Survived

#train.drop(['Survived'], axis = 1, inplace = True)



all_data = pd.concat([train, test])
percent_null_value(all_data)
print('all_data cabin unique value')

print(all_data['Cabin'].str[0].unique())
percent_unique_value(all_data, 'Cabin')
mapping = {'N': 0,

 'C': 1,

 'B': 2,

 'D': 3,

 'E': 4,

 'A': 5,

 'F': 6,

 'G': 6,

 'T': 6}



all_data['Cabin'] = all_data['Cabin'].map(mapping)
all_data['title'] = all_data['Name'].apply(lambda x : x.split('.')[0])

all_data['title'] = all_data['title'].apply(lambda x : x.split(' ')[-1])
percent_unique_value(all_data, 'title')
mapping = {'Mr': 0,

 'Miss': 1,

 'Mrs': 2,

 'Master': 3,

 'Dr': 4,

 'Rev': 4,

 'Col': 4,

 'Mlle': 4,

 'Major': 4,

 'Ms': 4,

 'Mme': 4,

 'Lady': 4,

 'Countess':4,

 'Don': 4,

 'Sir': 4,

 'Jonkheer': 4,

 'Capt': 4,

 'Dona': 4}



all_data['title'] = all_data['title'].map(mapping)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data[all_data.Survived == 0].groupby('FamilySize').agg({'Survived' : 'count'}).plot(kind = 'bar')

all_data[all_data.Survived == 1].groupby('FamilySize').agg({'Survived' : 'count'}).plot(kind = 'bar')
mapping = {1: 1, 2: 2, 3: 3, 4: 0, 6: 0, 5: 0, 7: 0, 11: 0, 8: 0}

all_data['FamilySize'] = all_data['FamilySize'].map(mapping)
all_data['is_alone'] = all_data['FamilySize'].apply(lambda x : 1 if x == 1 else 0)
percent_unique_value(all_data, 'Ticket')
alive = train[train.Survived == 1]

dead  = train[train.Survived == 0]



plt.figure(figsize = (15, 7))

sns.kdeplot(alive['Age'], shade = False, label = 'Survived')

sns.kdeplot(dead['Age'], shade = False, label = 'Dead')
all_data['Age'].fillna(all_data.groupby(['title'])['Age'].transform('median'), inplace = True)
def splitAgeColumns(x):

    if x <= 16.0:

        return 0

    elif x > 16.0 and x <= 26.0:

        return 1

    elif x > 26.0 and x <= 36.0:

        return 2

    elif x > 36.0 and x <= 46.0:

        return 3

    else:

        return 4
all_data['Age'] = all_data['Age'].apply(lambda x : splitAgeColumns(x))
all_data[all_data.Survived == 0]['Age'].value_counts().plot(kind = 'bar', label = 'dead')

plt.legend()

plt.show()

all_data[all_data.Survived == 1]['Age'].value_counts().plot(kind = 'bar', label = 'survived')

plt.legend()
percent_null_value(all_data)
alive = train[train.Survived == 1]

dead  = train[train.Survived == 0]



plt.figure(figsize = (15, 7))

sns.kdeplot(alive['Fare'], shade = False, label = 'Survived')

sns.kdeplot(dead['Fare'], shade = False, label = 'Dead')
all_data['calculate_fare'] = all_data['Fare'] / (all_data['Parch'] + all_data['SibSp'] + 1)
alive = all_data[all_data.Survived == 1]

dead  = all_data[all_data.Survived == 0]



plt.figure(figsize = (15, 7))

sns.kdeplot(alive['calculate_fare'], shade = True, label = 'Survived')

sns.kdeplot(dead['calculate_fare'], shade = True, label = 'Dead')

plt.xlim(0,60)

plt.ylim(0,.2)
def splitFareColumns(x):

    if x <= 5.0:

        return 0

    elif x > 5.0 and x <= 10.0:

        return 1

    elif x > 10.0 and x <= 15.0:

        return 2

    elif x > 15.0 and x <= 20.0:

        return 3

    elif x > 20.0 and x <= 25.0:

        return 4

    else:

        return 5
all_data['calculate_fare'] = all_data['calculate_fare'].apply(lambda x : splitFareColumns(x))
all_data.groupby('SibSp').agg({'Survived' : 'count'}).plot(kind = 'bar')
all_data.groupby('Parch').agg({'Survived' : 'count'}).plot(kind = 'bar')
def splitSibSpColumns(x):

    if x <= 0.0:

        return 0

    elif x > 0.0 and x <= 1.0:

        return 1

    else:

        return 2



def splitParchColumns(x):

    if x <= 0.0:

        return 0

    elif x > 0.0 and x <= 2.0:

        return 1

    else:

        return 2
all_data['SibSp'] = all_data['SibSp'].apply(lambda x : splitSibSpColumns(x))

all_data['Parch'] = all_data['Parch'].apply(lambda x : splitParchColumns(x))
pclass = [1, 2, 3]



for p_class in pclass:

    pclass_fare = train[train.Pclass == p_class]

    alive = pclass_fare[train.Survived == 1]

    dead  = pclass_fare[train.Survived == 0]



    plt.figure(figsize = (15, 7))

    sns.kdeplot(alive['Age'], shade = True, label = 'pclass_{}_age_alive'.format(p_class))

    sns.kdeplot(dead['Age'], shade = True, label = 'pclass_{}_age_dead'.format(p_class))

    plt.show()
pclass = [1, 2, 3]



for p_class in pclass:

    pclass_fare = train[train.Pclass == p_class]

    alive = pclass_fare[train.Survived == 1]

    dead  = pclass_fare[train.Survived == 0]



    plt.figure(figsize = (15, 7))

    sns.kdeplot(alive['Fare'], shade = True, label = 'pclass_{}_fare_alive'.format(p_class))

    sns.kdeplot(dead['Fare'], shade = True, label = 'pclass_{}_fare_alive'.format(p_class))

    plt.show()
data = [train, test]

for dataset in data:

    dataset['pclass_fare'] = dataset['Fare'] * dataset['Pclass']
percent_null_value(train)
all_data.columns
df = all_data.drop(['Name', 'Ticket', 'Survived', 'Fare'], axis = 1)
mapping = {'male' : 0, 'female' : 1}

df['Sex'] = df['Sex'].map(mapping)
def getOneHotEncode(df, col):

    cat = pd.get_dummies(df[col], prefix = col)

    df = pd.concat([df, cat], axis = 1)

    df.drop([col], axis = 1, inplace = True)

    return  df
one_hot_columns = ['Pclass', 'Cabin', 'Embarked', 'title']
df = df.sort_values('PassengerId')
mapping = {'S' : 0, 'C' : 1, 'Q' : 2}

df['Embarked'] = df['Embarked'].map(mapping)
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
train, test = df[df.PassengerId <= 891], df[df.PassengerId > 891]

train, test = train.drop(['PassengerId'], axis = 1), test.drop(['PassengerId'], axis = 1)



X_train, X_val, Y_train, Y_val = train_test_split(train, survived, test_size = 0.2, shuffle = True, random_state = 1)
clf = SVC(kernel = 'linear')

cross_val_score(clf, train, survived, cv = 5)
clf = XGBClassifier()

cross_val_score(clf, train, survived, cv = 5)
clf = DecisionTreeClassifier()

cross_val_score(clf, train, survived, cv = 5)
clf = RandomForestClassifier(n_estimators=400)

cross_val_score(clf, train, survived, cv = 5)
clf = CatBoostClassifier(verbose = False)

cross_val_score(clf, train, survived, cv = 5)
clf = SVC()

clf.fit(X_train, Y_train)
y_train_pred = clf.predict(X_train)

y_val_pred = clf.predict(X_val)

print('train accuracy: {}%'.format(accuracy_score(y_train_pred, Y_train)))

print('validation accuracy: {}%'.format(accuracy_score(y_val_pred, Y_val)))
y_test = clf.predict(test)

result = pd.read_csv('../input/titanic/gender_submission.csv')

result['Survived'] = y_test

result.to_csv('submission.csv', index = False)