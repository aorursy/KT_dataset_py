# Data Analysis libraries

import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Read the data into dataframes

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
# Combine train and test data

combine = [train, test]
train.head()
# Glance the data
train.describe(include='all')
# List the columns

train.columns.values
# Data info
train.info()
print('+-'*20)
test.info()
# Check for nulls
print(pd.isnull(train).sum())
# Bar plot - Sex vs Survived
sns.barplot(x="Sex", y="Survived", data= train)

# Printing percentages
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by="Survived", ascending=False)
# Barplot PClass vs Survived
sns.barplot(x="Pclass", y= "Survived", data=train)

#Printing percentages
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by="Survived", ascending=False)
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

# Printing Percentages
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)
#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()

train[["Parch","Survived"]].groupby(['Parch'],as_index=False).mean().sort_values(by="Survived",ascending=False)
print(train.columns.values)
# We want to extract the designation of names from the combined data set.
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
# Replace the titles with groups
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


train[['Title', 'Survived']].groupby(['Title'], as_index=False).sum()    
#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
print(pd.isnull(train).sum())
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

# Plot age vs survival

sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
#draw a bar plot for Parch vs. survival

train["Cabin_value"]= pd.notnull(train.loc[:,'Cabin']).astype(int)
print(train.columns.values)                             

test["Cabin_value"]= pd.notnull(test.loc[:,'Cabin']).astype(int)
print(test.columns.values) 
pd.crosstab(train['Cabin_value'], train['Pclass'])
pd.crosstab(train['Cabin_value'], train['Survived'])
sns.barplot(x="Cabin_value", y="Survived", data=train)
plt.show()

train[["Cabin_value","Survived"]].groupby(['Cabin_value'],as_index=False).mean().sort_values(by="Survived",ascending=False)
# See the test data
test.describe(include = "all")
# Check for nulls
print(pd.isnull(train).sum())
# Map age to a numerical value and drop the age feature
combine = [train, test]

title_mapping = {"Unknown": 0, "Baby": 1, "Child": 2, "Teenager": 3, "Student": 4, "Young Adult": 5, "Adult": 6, "Senior":7}
for dataset in combine:
    dataset['AgeGroup'] = dataset['AgeGroup'].map(title_mapping)
    dataset['AgeGroup'] = dataset['AgeGroup'].fillna(0)

train.head()
# Map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
combine = [train, test]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# fiilling with most occuring value through mode

freq_port = train.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
print(train.columns.values)
train = train.drop(["PassengerId","Name","Age","Ticket","SibSp","Parch","Fare","Cabin"], axis=1)
test = test.drop(["Name","Age","Ticket","SibSp","Parch","Fare","Cabin"], axis=1)
print(train.columns.values)
print(test.columns.values)
print(train.info())
print('+-'*20)
print(test.info())
test.head()
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log
# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_linear_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_linear_svc
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)
