# data analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import matplotlib.pyplot as plt # for data visualization

import seaborn as sns # for data visualization



sns.set_style('dark')



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load train data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# load test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print('='*50)

print("Number of columns in training data")

print('='*50)

print("\n")

print(train_data.columns.values)

print("\n")

print('='*50)

print("Number of columns in test data")

print('='*50)

print("\n")

print(test_data.columns.values)
print('='*10)

print("Train data shape")

print('='*10)

print("\n")

print(train_data.shape)

print("\n")

print('='*10)

print("Test data shape")

print('='*10)

print("\n")

print(test_data.shape)
print('='*50)

print("\nDescribe traing data\n")

print('='*50) 

print("\n")

print(train_data.describe())
print("Describe test data")

print('='*50)

print(test_data.describe())
print('='*50)

print("\nTraining data info\n")

print('='*50)

print(train_data.info())

print("\n")

print('='*50)

print("\n Test data info \n")

print('='*50)

print("\n")

print(test_data.info())
print('='*50)

print('\nNumber of null values in train data\n')

print('='*50)

print('\n')

print(train_data.isnull().sum())

print('\n')

print('='*50)

print('\n Number of null values in test data\n')

print('='*50)

print("\n")

print(test_data.isnull().sum())
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data = train_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)
train_data = train_data.drop(['Ticket'], axis = 1)

test_data = test_data.drop(['Ticket'], axis = 1)
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
# let check missing value again

print('='*50)

print('\nNumber of null values in train data\n')

print('='*50)

print('\n')

print(train_data.isnull().sum())

print('\n')

print('='*50)

print('\n Number of null values in test data\n')

print('='*50)

print("\n")

print(test_data.isnull().sum())
# number of survived passengers

train_data.groupby(['Survived'])['Survived'].count()
# percentage of male and female who survived

train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# percentage of people survived according to their Ticker Class

train_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Percentage of survived people based on their embarked. 

train_data[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x = 'Survived', data = train_data)
#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train_data)
#draw a bar plot of survival by sex

sns.barplot(x="Pclass", y="Survived", data=train_data)
#draw a bar plot of survival by sex

sns.barplot(x = "Embarked", y = "Survived", data = train_data)
#draw a bar plot of survival by sex

sns.barplot(x="Parch", y="Survived", data=train_data)
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')

grid.map(plt.hist, 'SibSp', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass')

grid.map(plt.hist, 'Embarked', alpha=.5, bins=20)

grid.add_legend();
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
train_data['Sex'] = train_data['Sex'].map({'male':1, 'female':0})

test_data['Sex'] = test_data['Sex'].map({'male':1, 'female':0})
train_data['Embarked'] = train_data['Embarked'].map({'Q':2, 'S':1, 'C':0})

test_data['Embarked'] = test_data['Embarked'].map({'Q':2, 'S':1, 'C':0})
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
X_train = train_data.drop(["Name", "Survived", "PassengerId"], axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop(['Name',"PassengerId"], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Support Vector Machine

svc = SVC()

svc.fit(X_train, Y_train)

svm_Y_pred = svc.predict(X_test)

svc_accuracy = svc.score(X_train, Y_train)

svc_accuracy
# k-nearest neighbor

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

knn_Y_pred = knn.predict(X_test)

knn_accuracy = knn.score(X_train, Y_train)

knn_accuracy
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

guassian_Y_pred = gaussian.predict(X_test)

gaussian_accuracy = gaussian.score(X_train, Y_train)

gaussian_accuracy
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

linear_svc_Y_pred = linear_svc.predict(X_test)

linear_svc_accuracy = linear_svc.score(X_train, Y_train)

linear_svc_accuracy
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

sgd_Y_pred = sgd.predict(X_test)

sgd_accuracy = sgd.score(X_train, Y_train)

sgd_accuracy
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

decision_tree_Y_pred = decision_tree.predict(X_test)

decision_tree_accuracy = decision_tree.score(X_train, Y_train)

decision_tree_accuracy
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest_Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

random_forest_accuracy = random_forest.score(X_train, Y_train)

random_forest_accuracy
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Gaussian Naive Bayes', 'Linear SVC',

              'Stochastic Gradient Decent', 'Decision Tree','Random Forest'],

    'Score': [svc_accuracy, knn_accuracy, gaussian_accuracy, linear_svc_accuracy, 

              sgd_accuracy, decision_tree_accuracy, random_forest_accuracy]})

models.sort_values(by='Score', ascending=False)
# submission file from each model

svm_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": svm_Y_pred})

svm_submission.to_csv('svm_submission.csv', index=False)



knn_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": knn_Y_pred})

knn_submission.to_csv('knn_submission.csv', index=False)



guassian_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": guassian_Y_pred})

guassian_submission.to_csv('guassian_submission.csv', index=False)



linear_svc_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": linear_svc_Y_pred})

linear_svc_submission.to_csv('linear_svc_submission.csv', index=False)



sgd_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": sgd_Y_pred})

sgd_submission.to_csv('sgd_submission.csv', index=False)



decision_tree_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": decision_tree_Y_pred})

decision_tree_submission.to_csv('decision_tree_submission.csv', index=False)



random_forest_submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": random_forest_Y_pred})

random_forest_submission.to_csv('random_forest_submission.csv', index=False)