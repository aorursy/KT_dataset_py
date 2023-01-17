#loading tests and train data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#data table
train.head()
#test data
test.head()
#shape of test and train data
train.shape

test.shape
#info about columns in test data
train.info()
#test set info
test.info()
#null data summary for train
train.isnull().sum()
#null data summary for test
test.isnull().sum()
#library for plotting data
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# setting seaborn default for plots
sns.set() 
#function to plot bar chart for survive with relation to other parameters
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')



bar_chart('Parch')
bar_chart('Embarked')
complete_data = [train,test] #combining both data
for dataset in complete_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand = False)
train['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in complete_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
bar_chart('Title')
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
train.head()
test.head()
sex_mapping = {"male":0,"female":1}
for dataset in complete_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
#filling missing age with median age of the title
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show() 
for dataset in complete_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
train.head()
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
for dataset in complete_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#mapping embarked
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in complete_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test.info()
train.info()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()  
for dataset in complete_data:
    dataset.loc[dataset['Fare'] <= 17,'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30),"Fare"] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 93),"Fare"] = 2
    dataset.loc[dataset['Fare'] > 93,"Fare"] = 3
train.head(20)
train.Cabin.value_counts()
#taking only the first character and eliminating the numbers
for dataset in complete_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
train.head()
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#mapping cabbin
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in complete_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing cabbin with median for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in complete_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head()
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape
test.head(20)
train.head()
X_train = train_data
Y_train = target
X_train.shape
Y_train.shape
X_test  = test.drop("PassengerId", axis=1).copy()
X_test.head()
X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 10)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# kNN Score
round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# decision tree Score
round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=11)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(score)*100, 2)
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Naive Bayes Score
round(np.mean(score)*100, 2)

clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#SVM score
round(np.mean(score)*100,2)
clf = SVC(gamma = 'scale')
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)