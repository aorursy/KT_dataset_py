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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.shape

test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
def barChar(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind = 'bar', stacked=True, figsize=(10,5))
barChar('Sex')
barChar('Pclass')
barChar('SibSp')
barChar('Parch')
train.isnull()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')
#0----> not survived
# 1---> survived

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
#see the distribution of age
#it helps us to see the avg age of people in titanic
sns.distplot(train['Age'].dropna(), kde=False, color='blue',bins=40)
#countplot of sibling 
sns.countplot(x='SibSp', data=train)
#average fare of ticket
train['Fare'].hist(color='red',bins=40)
train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data = train, palette='winter')
def inputAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
def embarkNull(cols):
    emb = cols
    if pd.isnull(emb):
        return 'S'
    
    else:
        return emb
# now apply this function
train['Age'] = train[['Age','Pclass']].apply(inputAge,axis=1)
train['Embarked'] = train['Embarked'].apply(embarkNull)
#again check heat map
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')
#again check heat map
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')
train.isnull().sum()
test.isnull().sum()

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data = test, palette='winter')
def inputAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 41
        elif Pclass == 2:
            return 27
        else:
            return 24
        
    else:
        return Age
def embarkNull(cols):
    emb = cols
    if pd.isnull(emb):
        return 'S'
    
    else:
        return emb
# now apply this function
test['Age'] = test[['Age','Pclass']].apply(inputAge,axis=1)

test['Embarked'] = test['Embarked'].apply(embarkNull)
#again check heat map
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')
train.head()
test.head()
train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
train.head()
test.head()
sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')
#fill missing fare
train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"),inplace=True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"),inplace=True)
sns.heatmap(test.isnull(),yticklabels=False, cbar=False,cmap='viridis')
train.isnull().sum()
test.isnull().sum()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()

sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
train.head()
test.head()
pd.crosstab(train['Title'],train['Sex'])
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
test.head()
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[(dataset['Age'] >32) & (dataset['Age'] <= 48), 'Age'] =2
    dataset.loc[(dataset['Age'] >48) & (dataset['Age'] <= 64), 'Age'] =3
    dataset.loc[dataset['Age'] >64, 'Age'] =4
train.head()
test.head()
train['FareBand'] = pd.qcut(train['Fare'], 4)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
train.Embarked.unique()

train.Embarked.value_counts()
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train.head(2)
train.columns
test.columns
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'FareBand'], axis=1)
train.head()

test.head()
x_train = train.drop('Survived', axis = 1)
y_train = train['Survived']

x_test = test.drop('PassengerId', axis=1).copy()
x_train.shape, y_train.shape, x_test.shape

#importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
x_train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accKNN = round(clf.score(x_train, y_train)*100, 2)
accKNN
clf = RandomForestClassifier(n_estimators=15)
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)
acc_RForest = round(np.mean(score)*100,2)
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)
acc_NB = round(np.mean(score)*100,2)
clf = SVC()
clf.fit(x_train, y_train)
y_pred_svc = clf.predict(x_test)
acc_svc = round(clf.score(x_train, y_train) * 100, 2)
print (acc_svc)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predDTree = clf.predict(x_test)
accDTree = round(clf.score(x_train,y_train) *100, 2)
print(accDTree)
clf = Perceptron(max_iter = 5, tol=None)
clf.fit(x_train,y_train)
y_predPercp = clf.predict(x_test)
accPercp = round(clf.score(x_train, y_train) * 100, 2)
print(accPercp)
models = pd.DataFrame({
    'Model':['KNN', 'Random Forest', 'Naive Bayes', 'SVM', 'Decision Tree','Perceptron'],
    'Score' : [accKNN, acc_RForest, acc_NB, acc_svc, accDTree, accPercp]
})
models.sort_values(by='Score', ascending=False)
test.head()
submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    "Survived" : y_predDTree
})
submission.to_csv('myResult.csv')