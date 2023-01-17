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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
train = pd.read_csv('/kaggle/input/titanic/train.csv', header=0, dtype={'Age': np.float64})
test = pd.read_csv('/kaggle/input/titanic/test.csv', header=0, dtype={'Age': np.float64})
full_data = [train, test]
train.info()
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print (train[['Pclass', "Survived"]].groupby(['Pclass'], as_index=False).mean())
# SibSp and Parch
for d in full_data:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

print (train[['FamilySize', "Survived"]].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
fp = train.Embarked.dropna().mode()[0]
fp
for d in full_data:
    d['Embarked'] = d['Embarked'].fillna('S')

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
#Fare
for d in full_data:
    d['Fare'] = d['Fare'].fillna(train['Fare'].median())
    
train['FareDivided'] = pd.qcut(train['Fare'], 4)

print(train[['FareDivided', 'Survived']].groupby(['FareDivided'], as_index=False).mean())
for d in full_data:
    avg_age = d['Age'].mean()
    std_age = d['Age'].std()
    age_null_count = d['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(avg_age - std_age, avg_age + std_age, size=age_null_count)
    d['Age'][np.isnan(d['Age'])] = age_null_random_list
    d['Age'] = d['Age'].astype(int)
    
train['AgeCategory'] = pd.cut(train['Age'], 5)

print (train[['AgeCategory', 'Survived']].groupby(['AgeCategory'], as_index=False).mean())
for d in full_data:
    d['Title'] = d.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for d in full_data:
    d['Title'] = d['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for d in full_data:
    d['Sex'] = d['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

print(train.head())
for d in full_data:
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    d['Title'] = d['Title'].map(title_mapping)
    d['Title'] = d['Title'].fillna(0)
    
    d['Embarked'] = d['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    d.loc[d['Fare'] <= 7.91, 'Fare'] = 0
    d.loc[(d['Fare'] > 7.91) & (d['Fare'] <= 14.454), 'Fare'] = 1
    d.loc[(d['Fare'] > 14.454) & (d['Fare'] <= 31), 'Fare']   = 2
    d.loc[d['Fare'] > 31, 'Fare'] = 3
    d['Fare'] = d['Fare'].astype(int)
    
    d.loc[d['Age'] <= 16, 'Age'] = 0
    d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age'] = 1
    d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
    d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
    d.loc[ d['Age'] > 64, 'Age'] = 4
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['AgeCategory', 'FareDivided'], axis=1)
train = train.drop('PassengerId', axis=1)

test  = test.drop(drop_elements, axis = 1)
train.head()
test.head()
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

X_test = test.drop('PassengerId', axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
def classify(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = round( clf.score(X_train, y_train) * 100, 2)
    print(str(acc) + ' percent')
classify(LogisticRegression())
classify(SVC())
classify(LinearSVC())
classify(KNeighborsClassifier(n_neighbors=5))
classify(DecisionTreeClassifier())
classify(RandomForestClassifier(n_estimators=200))
classify(GaussianNB())
classify(Perceptron(max_iter=5, tol=None))
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)
test.head()
subm = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived' : y_pred_decision_tree
})
subm.to_csv('Submission.csv', index=False)
submission = pd.read_csv('Submission.csv')
submission.head()