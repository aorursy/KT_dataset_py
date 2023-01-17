import numpy as np

import pandas as pd

from pandas import Series,DataFrame 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# loading training and testing data as a DataFrame

'''

train = pd.read_csv("../kaggle/train.csv")

test = pd.read_csv("../kaggle/test.csv")

'''

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# viewing top 5 instances of training data

train.head()
test.info()
print(train.columns.values)
train.tail()
# we can see which columns are numerical and which are catogerical

train.info()

print('.'*40)

test.info()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x="Sex", y="Survived", hue="Sex", data=train);

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);

sns.barplot(x="SibSp", y="Survived", data=train);

sns.barplot(x="Parch", y="Survived", data=train);

train['Age'].hist(bins=70)
train.info()

print('.'*40)

test.info()


def preprocessing(data):

    data['Title']=pd.Series(data.Name.str.extract(' ([A-Za-z]+)\.', expand=False))

    for eachone in data:

        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'other')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

    data = data.drop(['Name','PassengerId','Cabin','Ticket'],axis =1)

    data['Title'] = data['Title'].replace({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "other": 5})

    data['Sex'] = data['Sex'].replace({"female":1,"male":0})

    

    for eachone in data:

        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

        #filling Nan in age

        data['Age'] = data['Age'].fillna(data['Age'].mean())

        #filling Nan in Fare

        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

        

    data['Embarked'] = data['Embarked'].replace({"S":1,"C":2,"Q":3})

    print(set(data['Embarked']))

    return data
train_new = preprocessing(train)

test_new = preprocessing(test)
train_new.info()

print('.'*40)

test_new.info()
X_train = train_new.drop("Survived", axis=1)

Y_train = train_new["Survived"]

X_test  = test_new

X_train.shape, Y_train.shape, X_test.shape

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

acc_log = clf.score(X_train,Y_train)

print("Accuracy:",acc_log*100)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

acc_knn= clf.score(X_train,Y_train)

print("Accuracy:",acc_knn*100)
from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron,acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
