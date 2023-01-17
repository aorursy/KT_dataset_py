import numpy as np 

import pandas as pd

import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train_file = "/kaggle/input/titanic/train.csv"

data_test_file = "/kaggle/input/titanic/test.csv"



train = pd.read_csv(data_train_file)

test = pd.read_csv(data_test_file)
train.head()
test.head()
train.shape
test.shape
# explore the data type of each column

train.info()
#train.isnull().sum()
# Why do we care about the test data ?

# test.info()
train.describe().T
# Data distribution of numerical features

train.hist(bins = 10, figsize=(18, 16), color="#2c5af2");
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() 
def counter_plot(feature):

    plt.figure(figsize=(8,8))

    ax = sns.countplot(x=feature, hue="Survived", data=train, palette="Set2")

    ax.set_ylabel("# of Passengers")
counter_plot('Sex')
counter_plot('Pclass')
counter_plot('Embarked')
def bar_plot_survived_prob(feature):

    plt.figure(figsize=(8,8))

    ax = sns.barplot(x=feature, y="Survived", data=train, palette="Set2")

    ax.set_ylabel("Survived Probability")
bar_plot_survived_prob('Sex')
bar_plot_survived_prob('Pclass')
bar_plot_survived_prob('Embarked')
corr_matrix = train.corr()

corr_matrix



plt.figure(figsize=(12,8))

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap="YlGnBu")
train.head(10)
train.info()
test.info()
train['Age']=train['Age'].fillna(train['Age'].median())

test['Age']=test['Age'].fillna(test['Age'].median())
train['Cabin'] = train['Cabin'].fillna('U')

test['Cabin'] = test['Cabin'].fillna('U')
print(train['Embarked'].value_counts())

train['Embarked'].fillna('S',inplace=True)
test['Fare']=test['Fare'].fillna(test['Fare'].median())
train.head()
# Sex

# Use map to transfer string type to integer

train['Sex'] = train['Sex'].map({"male": 0, "female":1})

test['Sex'] = test['Sex'].map({"male": 0, "female":1})

train.head()
# Embarked

# Use map to transfer string type to integer

train['Embarked'] = train['Embarked'].map({"S": 1, "C": 2, "Q": 3})

test['Embarked'] = test['Embarked'].map({"S": 1, "C": 2, "Q": 3})

train.head()
# Cabin

train['Cabin'] = train['Cabin'].map(lambda x: x[0])

test['Cabin'] = test['Cabin'].map(lambda x: x[0])

train.head()
print(train['Cabin'].value_counts())
cabin_dummies = pd.get_dummies(train['Cabin'], prefix="Cabin")

cabin_dummies
# Cabin

train['Cabin'] = train['Cabin'].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, "T": 9 })

test['Cabin'] = test['Cabin'].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, "T": 9 })

train.head()
train.info()
test.info()
features_columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']

target_columns = ['Survived']



train_featues = train[features_columns]

train_target = train[target_columns]



test_featues = test[features_columns]
train_featues.head()
train_target
test_featues
from sklearn.model_selection import train_test_split

import eli5



X_train, X_test, y_train, y_test = train_test_split(train_featues, train_target, test_size=0.2, shuffle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train
# Logistic Regression

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
# feature importance

eli5.show_weights(clf)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
# feature importance

eli5.show_weights(clf)
# SVM

from sklearn.svm import SVC



clf = SVC()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors = 13)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
# feature importance

eli5.show_weights(clf)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# Logistic Regression

clf = LogisticRegression()

scoring = 'accuracy'

score = cross_val_score(clf, train_featues, train_target, cv=k_fold, n_jobs=1, scoring=scoring)

score

print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# Random Forest

clf = RandomForestClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_featues, train_target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# SVM

clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train_featues, train_target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# KNN

clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train_featues, train_target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

prediction = clf.predict(test_featues)
output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})

output.head(10)