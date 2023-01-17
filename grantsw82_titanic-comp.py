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

#I followed video here https://www.youtube.com/watch?v=COUWKVf6zKY to help walk me through concepts: https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
# Let's take a look at the data and load the train
titanic_train_file_path = '../input/train.csv'
train = pd.read_csv(titanic_train_file_path)
train.describe()

# load test data
titanic_test_file_path = '../input/test.csv'
test=pd.read_csv(titanic_test_file_path)
test.describe()
# look at data for train
train.head()

test.head()

#using .shape will tell you row and col
train.shape
test.shape
#.info() will tell you schema and row count
train.info()
# will find the nulls in the fields and sum them
train.isnull().sum()
# here is where i take a look at that data and see what trends I can find using bar charts
#This is good to critique the data early to see if you can gain insights on what variables
#to use
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() 
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked = True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
train.head()
#Feature Engineering


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
#Combine train and test to extract the title
train_test = [train,test]
for dataset in train_test:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
#Lets map the title to value for usin in algo
#Mr=0,Miss=1,Mrs=2,Others=3
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head()
bar_chart('Title')
train.isnull().sum()
#need to perform the same style mapping that was doen for the title but this time we will do it for sex
sex_mapping = {"male":0,
              "female":1}
for dataset in train_test:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()
bar_chart("Sex")
#Since Age is NUll in some areas, what we'll is use the median value of the age for the title
#to fill in the blanks using the .fillna function 
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace=True)
test["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace=True)

train.head()
#Lets plot it using a FacetGraph
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, train["Age"].max()))
facet.add_legend()
plt.show()
for dataset in train_test:
    dataset.loc[ dataset['Age'] <= 0, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 0) & (dataset['Age'] <= 5), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 12), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age'] = 3,
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 35), 'Age'] = 4,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 60), 'Age'] = 5,
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 6
bar_chart('Age')
#Going to take a look at Embarked

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#Since a majority of embarkation happened from S, Let's fill that out
for dataset in train_test:    
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
#Let map embarked now too!
embarked_mapping = {"S":0,
                   "C":1,
                   "Q":2}

for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#time to fill in Fare now
# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(50)
# use binning to classify the Fare
for dataset in train_test:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train.head()
#Last let looks at cabin
train.Cabin.value_counts()
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#DO the mapping

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
#Forgot about Family Size
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
#Map family
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head()
#time to cleanup
features_drop = ['Ticket']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
#train = train.drop(['PassengerId'], axis=1)
train_data = train.drop('Survived', axis=1)
target = train['Survived']


train_data.shape, target.shape
train_data.head(10)

#*******************MODELLING******************#

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
#need to splits data for cross validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

#kNN Scrore - find the average of the 10 that were returned
round(np.mean(score)*100,2)
#Lets check DecisionTree
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)
#RandomForest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(score)*100, 2)
#Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Naive Bayes Score
round(np.mean(score)*100, 2)
#SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#SVM score
round(np.mean(score)*100,2)
train_data.info()
#TESTING

clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head()
