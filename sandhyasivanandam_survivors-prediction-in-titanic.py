from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")
import pandas as pd

## Loading train and test data using pandas



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
## To check the features and the observations of train dataset



train.shape



## we have 891 observations and 12 features in our training dataset
## To print the basic statistical details of the dataframe

train.describe()
train.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting Seaborn for plots as default
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_chart('Pclass')

plt.title("Passenger Survival by Class")
bar_chart('Sex')

plt.title("Passenger Survival by Sex")
bar_chart('SibSp')

plt.title("Survival by SibSp")
bar_chart('Parch')

plt.title("Survival by Parch")
bar_chart('Embarked')

plt.title("Embarked Survival")
## bar_chart('Age') ## Legend is very long
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
train.head()
# Combine the train and test data

train_test_data = [train, test] 



# To extract the titles from the name feature(Mr, Mrs, Miss, Dr, Col and other titles)



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
train.head()
test.head()
title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, 

                 "Major":3, "Col":3, "Mlle":3, "Jonkheer":3,     

                  "Lady":3, "Mme":3, "Sir":3, "Don":3, "Ms":3, "Capt":3, "Countess":3, "Dona":3}



for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
# To check the mapped value of the feature Title in train data

train.tail()
# To check the mapped value of the feature Title in test data

test.tail()
## To visualise the data with respect to Title

bar_chart('Title')
# Dropping the unnecessary/irrelevant features from both train and test Dataset

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
train.head()
test.head()
Sex_mapping = {"male":0, "female":1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(Sex_mapping)
bar_chart('Sex')
train.head(100)
# Missing value of ages are filled with Median age for the titles (Mr, Mrs, Miss, others)

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"),inplace=True)
train.head()
train.groupby("Title")["Age"].transform("median")
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

 

plt.show() 
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(0, 20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(20,30)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(30,40)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(60,80)
# Combine the train and test data

train_test_data = [train, test] 



for dataset in train_test_data:

    dataset.loc[dataset['Age'] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[dataset['Age'] > 64,'Age']= 4
train.head()
test.head()
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
for dataset in train_test_data:

    dataset["Embarked"]=dataset['Embarked'].fillna('S')
train.head()
train.info()
embarked_mapping={"S": 0, "C": 1, "Q": 2 }

for dataset in train_test_data:

    dataset["Embarked"] = dataset['Embarked'].map(embarked_mapping)
train.head()
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(50)
train.info()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

 

plt.show()  
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(0, 20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(20,50)
# Combine the train and test data

train_test_data = [train, test] 



# Binning of the Fare with the range of values

for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train.head()
bar_chart('Fare')
train.info()
train.Cabin.value_counts()
# Combine the train and test data

train_test_data = [train, test] 



for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]

#print(train.Cabin.values)
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing Cabin with median Cabin value

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train.head()
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
# Binning of Family Size

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head()
test.head()
# Drop the unused columns

features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
train.head()
test.head(800)
# Drop the Survived column from train dataset

train_data = train.drop('Survived', axis=1)

target = train['Survived']



train_data.shape, target.shape
train_data.head(10)
# Importing Classifier Modules

# 5 Classifiers are mentioned below

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
train.info()
# Using the variable x_train and y_train for better clarity and assigning the train_data and target values to them

x_train = train_data

y_train = target
x_train.head()
y_train.head()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# n_splits is 10 as the data will be taken in 10 buckets and as it trains 10 times the accuracy will improve
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Decision Tree Score

round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Random Forest Score

round(np.mean(score)*100, 2)
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Gaussian NB Score

round(np.mean(score)*100, 2)
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# SVM Score

round(np.mean(score)*100, 2)
clf = SVC()

clf.fit(x_train, y_train)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })
print(submission)