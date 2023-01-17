#normal import

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



# visualization import

import seaborn as sb



# Disregard warning

import warnings 

warnings.filterwarnings('ignore')
test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
print(train.shape)
print(train.info())
train.head()
train.describe()
train.isnull().sum()
print(test.shape)
print(test.info())
test.head()
test.describe()
test.isnull().sum()
# For convinence, copy the data from original

train_survived = train[train['Survived'] == 1]

train_not_survived = train[train['Survived'] == 0]
train['Survived'].value_counts()
train['Sex'].value_counts()
train['Pclass'].value_counts()
train_survived.describe()
train_not_survived.describe()
age_survived = sb.FacetGrid(train, col = 'Survived')

age_survived.map(plt.hist, 'Age', bins = 20)
fare_survived = sb.FacetGrid(train, col = 'Survived')

fare_survived.map(plt.hist, 'Fare', bins = 20)
train_survived['Sex'].value_counts()
train_not_survived['Sex'].value_counts()
sb.barplot(x = 'SibSp', y = 'Survived', ci = None, data = train)
sb.barplot(x = 'Parch', y = 'Survived', ci = None, data = train)
sb.barplot(x = 'Pclass', y = 'Survived', ci = None, data = train)
sb.barplot(x = 'Embarked', y = 'Survived', ci = None, data = train)
# Classification module

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
drop_features = ['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp']

train_fixed = train.drop(drop_features, axis = 1)

test_fixed = test.drop(drop_features, axis = 1)
train_fixed['Age'].fillna(train['Age'].mean(), inplace = True)

test_fixed['Age'].fillna(train['Age'].mean(), inplace = True)
train_fixed['Sex'] = train_fixed['Sex'].map({'male' : 0, 'female' : 1}).astype(int)

test_fixed['Sex'] = test_fixed['Sex'].map({'male' : 0, 'female' : 1}).astype(int)
test_fixed['Fare'].fillna(train['Fare'].mean(), inplace = True)
train_fixed['Embarked'].fillna('S', inplace = True)
train_fixed['Embarked'] = train_fixed['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)

test_fixed['Embarked'] = test_fixed['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)
#Age grouping

mean_age = train_fixed['Age'].mean()

std_age = train_fixed['Age'].std()

max_age = train_fixed['Age'].max()



age_range = [0, mean_age - std_age, mean_age, mean_age + std_age, max_age]

level = [0, 1, 2, 3]

train_fixed['AgeRange'] = pd.cut(train_fixed['Age'], bins = age_range, labels = level)

test_fixed['AgeRange'] = pd.cut(test_fixed['Age'], bins = age_range, labels = level)
#Fare grouping

train_fixed['Fareband'] = pd.qcut(train_fixed['Fare'], 4)

train_fixed['Fareband'].value_counts()
train_fixed = train_fixed.drop('Fareband', axis = 1)
fare_range = [-0.001, 7.91, 14.454, 31.0, 513]

fare_level = [0, 1, 2, 3]

train_fixed['FareRange'] = pd.cut(train_fixed['Fare'], bins = fare_range, labels = fare_level)

test_fixed['FareRange'] = pd.cut(test_fixed['Fare'], bins = fare_range, labels = fare_level)
train_fixed['ParchTF'] = train_fixed['Parch'].map({1 : 0, 2 : 0, 3: 0, 0 : 0, 4 : 1, 5 : 1, 6 : 1}).astype(int)

test_fixed['ParchTF'] = test_fixed['Parch'].map({1 : 0, 2 : 0, 3: 0, 0 : 0, 4 : 1, 5 : 1, 6 : 1, 9 : 1}).astype(int)
# Delete Age, Fare, and Parch 

drop_age_fare = ['Age', 'Fare', 'Parch']

train_fixed = train_fixed.drop(drop_age_fare, axis = 1)

test_fixed = test_fixed.drop(drop_age_fare, axis = 1)
X_train = train_fixed.drop('Survived', axis = 1)

Y_train = train_fixed['Survived']

X_test = test_fixed

X_test = X_test.dropna()
logis_classify = LogisticRegression()

logis_classify.fit(X_train, Y_train)

Y_prediction = logis_classify.predict(X_test)

logis_score = logis_classify.score(X_train, Y_train)

print("The score of Logistic Regression is : " + str(logis_score))
svm_classify = SVC()

svm_classify.fit(X_train, Y_train)

Y_prediction_svm = svm_classify.predict(X_test)

svm_score = svm_classify.score(X_train, Y_train)

print("The score of SVM is : " + str(svm_score))
knn_classify = KNeighborsClassifier(n_neighbors = 2)

knn_classify.fit(X_train, Y_train)

Y_prediction_knn = knn_classify.predict(X_test)

knn_score = knn_classify.score(X_train, Y_train)

print("The score of KNN is : " + str(knn_score))
tree_classify = DecisionTreeClassifier()

tree_classify.fit(X_train, Y_train)

Y_prediction_tree = tree_classify.predict(X_test)

tree_score = tree_classify.score(X_train, Y_train)

print("The score of Decision tree is : " + str(tree_score))
forest_classify = RandomForestClassifier()

forest_classify.fit(X_train, Y_train)

Y_prediction_forest = forest_classify.predict(X_test)

forest_score = forest_classify.score(X_train, Y_train)

print("The score of Random Forest is : " + str(forest_score))
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Y_prediction_tree})

submission.to_csv('submission.csv', index=False)