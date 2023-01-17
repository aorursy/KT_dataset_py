# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load data
trainDf = pd.read_csv('../input/train.csv')
testDf = pd.read_csv('../input/test.csv')

#Dropping features that don't seem to contribute to the survival chance
trainDf = trainDf.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
testDf = testDf.drop(['Ticket', 'Cabin'], axis=1)

combine = [trainDf, testDf]

#Extract titles
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(', ([A-Za-z]+)\.')
pd.crosstab(trainDf['Title'], trainDf['Sex'])
#Replace rare titles with string "Rare"
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Lady',
                                                'Major', 'Mlle', 'Mme', 'Ms', 'Rev', 'Sir',
                                                'Dr'], 'Rare')
trainDf['Title'].unique()
#Convert categorical titles to ordinal
title_mapping = {"Mr": 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype('int64')

#Drop 'Name' feature from the dataset
trainDf = trainDf.drop(['Name'], axis=1)
testDf = testDf.drop(['Name'], axis=1)    
combine = [trainDf, testDf]

#Convert Sex feature to numerical value 0/1
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1})

guess_age = np.zeros([2, 3])
#Replace missing value of age based on title and sex
#2 possible values of Sex
for dataset in combine:
    for i in range(0, 2):
        #3 possible values of class
        for j in range(0, 3):
            guess_df = dataset.loc[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1), ['Age']].dropna()
            guess_age[i, j] = guess_df.mean()
            dataset.loc[(dataset['Sex'] == i) & \
                       (dataset['Pclass'] == j+1) &\
                       dataset['Age'].isnull(), ['Age']] = guess_age[i, j]
    dataset['Age'] = dataset['Age'].astype('int64')
#Create age band
trainDf['Ageband'] = pd.cut(trainDf['Age'], 5)
trainDf[['Ageband', 'Survived']].groupby('Ageband').mean()

#Replace Age with ordinals
for dataset in combine:
    dataset.loc[dataset['Age'] < 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] >= 16) & (dataset['Age'] < 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] >= 32) & (dataset['Age'] < 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] >= 48) & (dataset['Age'] < 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] >= 64) & (dataset['Age'] < 80), 'Age'] = 4
    dataset.loc[dataset['Age'] > 80, 'Age'] = 5

#Create fair band
trainDf['Fareband'] = pd.qcut(trainDf['Fare'], 4)
trainDf[['Fareband', 'Survived']].groupby('Fareband').mean()

#Replace fare and embarked with ordinal
for dataset in combine:
    dataset.loc[(dataset['Fare'] < 8), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] >= 8) & (dataset['Fare'] < 14), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] >= 14) & (dataset['Fare'] < 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] >= 31) & (dataset['Fare'] < 513), 'Fare'] = 3
    dataset.loc[(dataset['Embarked'] == 'S'), 'Embarked'] = 0
    dataset.loc[(dataset['Embarked'] == 'C'), 'Embarked'] = 1
    dataset.loc[(dataset['Embarked'] == 'Q'), 'Embarked'] = 2
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
    dataset['Fare'] = dataset['Fare'].fillna(0)
trainDf = trainDf.drop(['Ageband', 'Fareband'], axis=1)
#Training data
X_train = trainDf.drop('Survived', axis=1)
Y_train = trainDf['Survived']
X_test = testDf.drop('PassengerId', axis=1)
X_train.shape, Y_train.shape, X_test.shape


from sklearn.linear_model import LogisticRegression
#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc = logreg.score(X_train, Y_train)
#Random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
acc = random_forest.score(X_train, Y_train)
print(acc)
Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({
            'PassengerId': testDf['PassengerId'],
            'Survived': Y_pred
            })
submission.to_csv('submission.csv', index=False)
