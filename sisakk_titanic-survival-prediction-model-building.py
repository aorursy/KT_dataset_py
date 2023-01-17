# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from math import ceil # import ceil function from math module
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# model building
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe(include='all')
train.info()
train.sample(5)
# get total and percent of missing values for each feature
total_missing = train.isnull().sum().sort_values(ascending=False)
percent_missing = (train.isnull().sum() * 100/train.isnull().count()).sort_values(ascending=False)
pd.concat([total_missing, percent_missing], axis=1, keys=['total', 'percent'])
print(train.groupby('Sex')['Survived'].mean())
# train.plot(x='Sex', y='Survived')
# train.groupby('Sex')['Survived'].mean().plot(x='Sex')

sns.barplot(x='Sex', y='Survived', data=train).set_title('Percent Survived by Gender')
# percent survived by pclass
sns.barplot(x='Pclass', y='Survived', data=train)

#print percentage of people by Pclass that survived
print("Pclass 1:", round(train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100, 2), "% survived")
print("Pclass 2:", round(train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100, 2), '% survived')
print("Pclass 3:", round(train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100, 2), '% survived')
# passengers survived by Pclass and sex
sns.barplot(x='Pclass', y='Survived', data=train, hue='Sex')
sns.barplot(x='Pclass', y='Survived', data=train, hue='Embarked')
# import math
train_nonNullAge = train[train['Age'].notnull()]
train_nonNullAge['Age'] = train_nonNullAge['Age'].apply(lambda x: ceil(x))

plt.title('Survival rate across ages ( 0 to 1 )')
train_nonNullAge.groupby('Age')['Survived'].mean().plot(kind='line')
# Combine Sibsp, Parch into Family feature and drop the 2 features
train['FamilySize'] = train['SibSp'] + train['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# drop Cabin, ticket and name of passenger
train.drop(['Cabin','Name', 'Ticket'], axis=1, inplace=True)
# Remove rows with null embarked values
train.drop(train[train['Embarked'].isnull()].index, inplace=True)
# Fill age values in null fields using randint(mean-std, mean+std, count(null values))
train['Age'].fillna(value=np.random.randint(train['Age'].mean() - train['Age'].std(), train['Age'].mean() + train['Age'].std()), inplace=True)
train.isnull().any().any()
# show max 11 columns
pd.set_option('display.max_columns', 11)

# convert categoricals sex and embarked to numericals for modelling purpose

# lets map sex male to 0, female to 1
sex_map = {'female': 0, 'male': 1}

# Emabrked mapping
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

train['Sex'] = train['Sex'].map(sex_map)
train['Embarked'] = train['Embarked'].map(embarked_map)

train.head()
# from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
# Comparing accuracies of each models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
# Clean test data and Submit Predictions
test.describe(include='all')
# missing values for test data
total_missing_test = test.isnull().sum().sort_values(ascending=False)
percent_missing_test = (test.isnull().sum() * 100/test.isnull().count()).sort_values(ascending=False)
pd.concat([total_missing_test, percent_missing_test], axis=1, keys=['total', 'percent'])
# save passenger IDs to use in submission file
test_ids = test['PassengerId']

# drop cabin, Name, Ticket, combine Sibsp, Parch into FamiliSize and drop those 2 features
test['FamilySize'] = test['SibSp'] + test['Parch']
test.drop(['PassengerId', 'Cabin', 'SibSp', 'Parch', 'Name', 'Ticket'], axis=1, inplace=True)

test.head()
# fill with mean values for fare
test['Age'].fillna(value=np.random.randint(test['Age'].mean() - test['Age'].std(), test['Age'].mean() + test['Age'].std()), inplace=True)
test.isnull().sum()
# null record of Fare in test
test_fare_null = test[test['Fare'].isnull()].index
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# False implying that no feature of any record is null.
test.isnull().any().any()
# converting test categoricals sex and embarked to numericals just like we did for train

# sex_map = {'female': 0, 'male': 1}

# Emabrked mapping
# embarked_map = {'S': 0, 'C': 1, 'Q': 2}

test['Sex'] = test['Sex'].map(sex_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

# all data is converted to format the way we did for train data.
# test data is ready to be fed into random forrest model since it gave maximum accuracy.
test.head()
# randomforest

#set the output as a dataframe and convert to csv file named submission.csv
submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": randomforest.predict(test)
    })

submission.to_csv('submission.csv', index=False)