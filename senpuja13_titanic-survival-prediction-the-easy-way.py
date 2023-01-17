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
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train_set.head(5)
train_set.shape
train_set.info()
test_set.head(5)
test_set.shape
test_set.info()
train_set.head(5)
train_test_dataset = [train_set, test_set]
for dataset in train_test_dataset:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_set['Title'].value_counts()
test_set['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4,
                 "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }
for dataset in train_test_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train_set.head(5)
test_set.head(5)
train_set.drop('Name', axis = 1, inplace = True)
test_set.drop('Name', axis = 1, inplace = True)
train_set.head(5)
test_set.head(5)
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_dataset:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train_set.head(5)
test_set.head(5)
train_set['Age'].fillna(train_set.groupby("Title")['Age'].transform("median"), inplace = True)
test_set['Age'].fillna(test_set.groupby("Title")['Age'].transform("median"), inplace = True)
train_set.head(20)
for dataset in train_test_dataset:
    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 20), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 35), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 50), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4
train_set.head(5)
Pclass1 = train_set[train_set['Pclass']==1]['Embarked'].value_counts()
Pclass1
Pclass2 = train_set[train_set['Pclass']==2]['Embarked'].value_counts()
Pclass2
Pclass3 = train_set[train_set['Pclass']==3]['Embarked'].value_counts()
Pclass3
for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train_set.head(5)
train_set["Family"] = train_set["SibSp"] + train_set["Parch"] + 1
test_set["Family"] = test_set["SibSp"] + test_set["Parch"] + 1
train_set.drop('Parch', axis = 1, inplace = True)
test_set.drop('Parch', axis = 1, inplace = True)
train_set.drop('SibSp', axis = 1, inplace = True)
test_set.drop('SibSp', axis = 1, inplace = True)
train_set.head(10)
train_set['Family'].value_counts()
test_set['Family'].value_counts()
family_mapping = {1: 0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1, 7: 1.2, 8: 1.4, 9: 1.6, 10: 1.8, 11: 2}
for dataset in train_test_dataset:
    dataset['Family'] = dataset['Family'].map(family_mapping)
train_set.head(5)
train_set["Fare"].fillna(train_set.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_set["Fare"].fillna(test_set.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_dataset:
    dataset.loc[ dataset['Fare'] <= 15, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train_set.head(5)
train_set['Cabin'].value_counts()
Pclass1 = train_set[train_set['Pclass']==1]['Cabin'].value_counts()
Pclass1
Pclass2 = train_set[train_set['Pclass']==2]['Cabin'].value_counts()
Pclass2
Pclass3 = train_set[train_set['Pclass']==3]['Cabin'].value_counts()
Pclass3
for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8, "F": 1, "G": 1.2, "T": 1.4}
for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train_set
train_set["Cabin"].fillna(train_set.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_set["Cabin"].fillna(test_set.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train_set.drop('Ticket', axis = 1, inplace = True)
test_set.drop('Ticket', axis = 1, inplace = True)
train_set.drop('PassengerId', axis = 1, inplace = True)
train_data = train_set.drop('Survived', axis=1)
target = train_set['Survived']

train_data.shape, target.shape
train_data
train_data.info()
from sklearn.svm import SVC
clfr = SVC()
clfr.fit(train_data, target)

test_data = test_set.drop("PassengerId", axis=1).copy()
prediction = clfr.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head()