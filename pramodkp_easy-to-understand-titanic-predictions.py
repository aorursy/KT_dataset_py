# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.info()
train.describe()
train.isnull().sum()
test.info()
test.describe()
test.isnull().sum()
num_of_survided = train[train['Survived'] == 1]

num_of_nonSurvided = train[train['Survived'] == 0]

print("Survided :", len(num_of_survided))

print("Not Survided :", len(num_of_nonSurvided))

print("Total Passangers :", len(train))
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
train.groupby('Pclass').Survived.value_counts()
sns.distplot(train['Age'].dropna(), kde=False, bins=30)
train['Age'].dropna().mean()
train.groupby('SibSp').Survived.value_counts()
sns.countplot(x='SibSp', data=train)
sns.countplot(x='SibSp', hue='Survived', data=train)
train['Fare'].hist(bins=30)
sns.countplot(x='Parch', hue='Survived', data=train)
train.groupby('Embarked').Survived.value_counts()
sns.countplot(x='Embarked', hue='Survived', data=train)
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)
train.isnull().sum()
sns.boxplot(x='Pclass',y='Age',data=train)
train[train['Pclass'] == 3]['Age'].mean()
def fillAge(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return round(train[train['Pclass'] == 1]['Age'].mean())

        elif Pclass == 2:

            return round(train[train['Pclass'] == 2]['Age'].mean())

        elif Pclass == 3:

            return round(train[train['Pclass'] == 3]['Age'].mean())

    else:

        return Age
train['Age'] = train[['Age', 'Pclass']].apply(fillAge, axis=1)
def TestfillAge(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return round(test[test['Pclass'] == 1]['Age'].mean())

        elif Pclass == 2:

            return round(test[test['Pclass'] == 2]['Age'].mean())

        elif Pclass == 3:

            return round(test[test['Pclass'] == 3]['Age'].mean())

    else:

        return Age
test['Age'] = test[['Age', 'Pclass']].apply(TestfillAge, axis=1)
train.isnull().sum()
train.head()
sex = pd.get_dummies(train['Sex'], drop_first=True)

embark = pd.get_dummies(train['Embarked'], drop_first=True)
testSex = pd.get_dummies(test['Sex'], drop_first=True)

testEmbark = pd.get_dummies(test['Embarked'], drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
test = pd.concat([test,testSex,testEmbark],axis=1)
train.head(2)

test.head(2)
train.drop('PassengerId',axis=1,inplace=True)
test.fillna(test['Fare'].mean(), inplace=True)
train.head(2)
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
logReg = LogisticRegression()

logReg.fit(X_train, y_train)

y_pred_log_reg = logReg.predict(X_test)

accuracy_log_reg = round(logReg.score(X_train, y_train) * 100, 2)

accuracy_log_reg
linear_svm = LinearSVC()

linear_svm.fit(X_train, y_train)

y_pred_linear_svm = linear_svm.predict(X_test)

accuracy_linear_svm = round(linear_svm.score(X_train, y_train) * 100, 2)

accuracy_linear_svm
svm = SVC()

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

accuracy_svm = round(svm.score(X_train, y_train) * 100, 2)

accuracy_svm
dec_tree = DecisionTreeClassifier()

dec_tree.fit(X_train, y_train)

y_pred_decision_tree = dec_tree.predict(X_test)

accuracy_decision_tree = round(dec_tree.score(X_train, y_train) * 100, 2)

accuracy_decision_tree
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = round(knn.score(X_train, y_train) * 100, 2)

accuracy_knn
nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

accuracy_nb = round(nb.score(X_train, y_train) * 100, 2)

accuracy_nb
rand_forest = RandomForestClassifier(n_estimators=50)

rand_forest.fit(X_train, y_train)

y_pred_random_forest = rand_forest.predict(X_test)

accuracy_random_forest = round(rand_forest.score(X_train, y_train) * 100, 2)

accuracy_random_forest
model_accuracy_list = pd.DataFrame({

    'Claasifier' : ['Logistic Regression', 'Linear SVC', 'SVC', 

                    'Decision Trees', 'KNN', 'Naive Bayes', 'Random Forect'],

    'Prediction Accuracy' : [accuracy_log_reg, accuracy_linear_svm, accuracy_svm,

                   accuracy_decision_tree, accuracy_knn, accuracy_nb, accuracy_random_forest]

})
model_accuracy_list
test.info()
submission = pd.DataFrame({

    "PassengerId" : test['PassengerId'],

    "Survived" : y_pred_random_forest # Any other classifier predictions

})
#submission.to_csv('submission5.csv', index=False)