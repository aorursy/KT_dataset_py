#data analysis libraries



import numpy as np

import pandas as pd



#visualization libraries



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')
#train data

url = '../input/titanic/train.csv'

train = pd.read_csv(url)



url = '../input/titanic/test.csv'

test = pd.read_csv(url)
train.describe(include="all")
#show the first ten rows of the dataset

train.sample(10)
print(pd.isnull(train).sum())
#barplot of survivals by sex

sns.barplot(x="Sex", y="Survived", data=train)
#Barplot of Pclass.

sns.barplot(x="Pclass", y="Survived", data=train)
train.drop(['PassengerId'], 1).hist(figsize=(25,18))

plt.show()
print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
#Filling the missing values with S (where the majority embarked).

train = train.fillna({"Embarked": "S"})
test = test.fillna({"Embarked": "S"})
#Filling the missing values based on mean fare for that Pclass

#PS. Only the test database has null values

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

    
#converting to numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
#Drop Ticket and Cabin

train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)
#And fill the missing Age values

train['Age'] = train.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))

train['Age'] = train.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

train['Age'] = train.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
test['Age'] = test.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))

test['Age'] = test.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

test['Age'] = test.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
#count the missing values of the Age feature

print(pd.isnull(train['Age']).sum())
print(pd.isnull(test['Age']).sum())
#Here we will create a Title column

train['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train['Name']), index=train.index)

train['Title'] = train['Title'].replace(['Lady','the Countess','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

train['Title'] = train['Title'].replace(['Mlle','Ms'], 'Miss')

train['Title'] = train['Title'].replace('Mme','Mrs')

train['Title'] = train['Title'].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5})



test['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in test['Name']), index=test.index)

test['Title'] = test['Title'].replace(['Lady','the Countess','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Mlle','Ms'], 'Miss')

test['Title'] = test['Title'].replace('Mme','Mrs')

test['Title'] = test['Title'].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5})
#Drop name and PassengerId

train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)
#converting sex feature to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)



test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#same to Embarked feature

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)



test['Embarked'] = test['Embarked'].map(embarked_mapping)
plt.subplots(figsize = (12, 12))

sns.heatmap(train.corr(), annot = True, linewidths = .5)
train.head()
#splitting training data to test accuracy.

from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived'],axis = 1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
#KNN

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
#Logistic Reggression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
#Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
#Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
#Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
#Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
#Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)




models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 'SVC',  

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron, acc_linear_svc, acc_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)



ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis = 1))

submission = pd.DataFrame({'PassengerId' : ids, 'Survived' : predictions})

submission.to_csv('submission.csv', index = False)