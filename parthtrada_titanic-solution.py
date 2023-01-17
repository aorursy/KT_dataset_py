# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic-solution-for-beginners-guide/train.csv")

test = pd.read_csv("/kaggle/input/titanic-solution-for-beginners-guide/test.csv")
train.isnull().sum()
test.isnull().sum()
train_test_data = [train , test ]   # concate the train and split data
train['Age'].fillna((train['Age'].mean()), inplace=True)

test['Age'].fillna((test['Age'].mean()), inplace=True)

test['Fare'].fillna((test['Fare'].mean()), inplace=True)

train['Embarked'].fillna((train['Embarked'].ffill()) , inplace=True)
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
train['Cabin'].fillna((train['Cabin'].ffill()) , inplace=True)

test['Cabin'].fillna((test['Cabin'].ffill()) , inplace=True)
train['Cabin'].fillna((train['Cabin'].bfill()) , inplace=True)

test['Cabin'].fillna((test['Cabin'].bfill()) , inplace=True)
train.isnull().sum()
test.isnull().sum()
train.head(5)
for dataset in train_test_data:

    dataset.loc[ dataset['Sex'] == 'male', 'Sex'] = 0,

    dataset.loc[ dataset['Sex'] == 'female', 'Sex'] = 1


train.Embarked.value_counts()
test.Embarked.value_counts()
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train.head(5)
train_test_data = [train, test] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
train.head(5)
features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
train_dat = train.drop('Survived', axis=1)

target = train['Survived']



train_dat.shape, target.shape
# modelling



from sklearn.svm import SVC



import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train_dat, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
clf = SVC()

clf.fit(train_dat, target)



test_dat = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_dat)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()