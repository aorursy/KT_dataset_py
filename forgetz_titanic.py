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

import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
combine = [train_df, test_df]
train_df.shape, test_df.shape
test_df.head()
train_df.describe()
train_df.columns.values
train_df.isnull().any()
pclass_survived = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot('Pclass', 'Survived', data = pclass_survived)
sex_survive = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sns.barplot('Sex', 'Survived', data = sex_survive)
sns.barplot('Pclass', 'Survived', hue='Sex', data = train_df)
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
sex_mapping = {"male": 0, "female": 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    dataset['Sex'] = dataset['Sex'].fillna(0)

train_df.head()
for dataset in combine:
    dataset['Age'] = dataset['Age'].replace(np.nan, dataset['Age'].mean(), regex=True)
    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
frequence_port = train_df.Embarked.dropna().mode()[0]

'''
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].replace(np.nan, "M", regex=True)
'''
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(frequence_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
#embark_mapping = {"S": 0, "C": 1, "Q": 2, "M": 3}

embark_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping).astype(int)

train_df.head()
test_df.isnull().any()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
test_df.head(10)
train_df.head(10)
predict_train = train_df.drop(["Survived", "Name"], axis=1)
predict_test = test_df.drop(["PassengerId", "Name"], axis=1)
X_train = predict_train
Y_train = train_df["Survived"]
X_test  = predict_test
X_train.shape, Y_train.shape, X_test.shape
X_test.head()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_dct_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_knn_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_rf_pred = rf.predict(X_test)
acc_rf = round(rf.score(X_train, Y_train) * 100, 2)
acc_rf
submission = pd.read_csv('../input/gender_submission.csv')
temp = pd.DataFrame({"0-PassengerId": test_df["PassengerId"], 
                     "1-Submission": submission['Survived'], 
                     "2-DecissionTree": Y_dct_pred,
                     "3-KNN": Y_knn_pred,
                     "4-RandomForest": Y_rf_pred
                    })
temp.head(20)
