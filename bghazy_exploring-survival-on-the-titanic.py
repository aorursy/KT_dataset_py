# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization

# machine learning

from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full_data = [train, test]

train.set_index('PassengerId', inplace=True, drop=True) #add PassengerId as index to train dataset

test.set_index('PassengerId', inplace=True, drop=True) #add PassengerId as index to train dataset

print (train.info()) #

train.head(3)
survived = train[train.Survived ==1]

dead = train[train.Survived ==0]



def plot_hist(feature, bins =20):

    x1 = np.array(dead[feature].dropna())

    x2 = np.array(survived[feature].dropna())

    plt.hist([x1, x2], label = ['Victime', 'Survivant'], bins= bins)#, color = ['', 'b'])

    plt.legend( loc = 'upper left')

    plt.title('distribution relative de %s ' %feature)

    plt.show()
plot_hist('Pclass')

print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
plot_hist('Age')

print (train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean())
plot_hist('SibSp')

print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
plot_hist('Parch')

print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 2} ).astype(int)

    

    # Mapping is_child (<18 years)

    dataset.loc[ dataset['Age'] < 18, 'is_child'] = 1

    dataset.loc[ dataset['Age'] >= 18, 'is_child'] = 0

    

    #Mapping is_alone (SibSp and Parch == 0)

    dataset.loc[ dataset['Parch'] + dataset['SibSp'] == 0, 'is_alone'] = 1

    dataset.loc[ dataset['Parch'] + dataset['SibSp'] > 0, 'is_alone'] = 0

# Feature Selection

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'Fare', 'Age', 'Embarked']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)

train.dropna(subset=['is_child'], inplace=True)

test.dropna(subset=['is_child'], inplace=True)

print (train.head(10))
def parse_model(X):

    target = X['Survived']

    X = X [['Sex', 'is_child', 'is_alone', 'Pclass']]

    return X, target

X, y = parse_model(train.copy())
def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv=5)

    return np.mean(xval)
lr =LogisticRegression()

compute_score(lr, X, y)
rf = RandomForestClassifier()

compute_score(rf, X, y)
rf.fit(X, y)

result = rf.predict(test)