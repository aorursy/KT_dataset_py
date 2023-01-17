# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

full_data = [df_train,df_test]
df_train.describe()
df_train["Sex"]=df_train["Sex"].map({"male":1,"female":0})

df_test["Sex"]=df_test["Sex"].map({"male":1,"female":0})
df_train.corr()**2
class_sur = df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()

class_sur.plot.bar(x="Pclass",y="Survived")
sex_sur = df_train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean()

sex_sur.plot(x="Sex",y="Survived",kind="bar")
df_train['Embarked']=df_train['Embarked'].map({"C":1,"Q":2,"S":3})

df_test['Embarked']=df_test['Embarked'].map({"C":1,"Q":2,"S":3})

df_train.head()                                  
emb_sur = df_train[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean()

emb_sur.plot(x="Embarked",y="Survived",kind="bar")
df_train['Family_members'] = df_train['SibSp'] + df_train['Parch']

df_test['Family_members'] = df_test['SibSp'] + df_test['Parch']

fam_mem_sur = df_train[['Family_members','Survived']].groupby(['Family_members'],as_index=False).mean()

fam_mem_sur.plot(x="Family_members",y="Survived",kind="bar")
df_train = df_train.drop(['PassengerId'],axis=1)

full_data = [df_train,df_test]
for dataset in full_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Vip')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    

title_sur =df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_sur.plot(x="Title",y="Survived",kind="bar")
for dataset in full_data:

    grouped = dataset.groupby(["Sex","Pclass",'Title'])

    dataset["Age"] = grouped["Age"].apply(lambda x: x.fillna(x.median()))

    dataset["Cabin"] = dataset["Cabin"].fillna("U")

    dataset["Cabin"] =  dataset["Cabin"].map(lambda x: x[0])

    most_emb = dataset["Embarked"].mode()[0]

    dataset["Embarked"] = dataset["Embarked"].fillna(most_emb)

    dataset["Fare"] =dataset["Fare"].fillna(dataset["Fare"].median())
df_train = df_train.drop(["Name",'SibSp','Parch','Fare',"Ticket"],axis = 1)

df_test = df_test.drop(["PassengerId","Name",'SibSp','Parch','Fare',"Ticket"],axis=1)
X_train = pd.get_dummies(df_train, columns=['Pclass', 'Sex',"Cabin", 'Embarked',

       'Family_members', 'Title'], prefix=['Pclass', 'Sex',"Cabin", 'Embarked',

       'Family_members', 'Title'])

Y_train = X_train['Survived']

X_train = X_train.drop(['Survived',"Cabin_T"],axis=1)

X_test = pd.get_dummies(df_test, columns=['Pclass', 'Sex',"Cabin", 'Embarked',

       'Family_members', 'Title'], prefix=['Pclass', 'Sex',"Cabin", 'Embarked',

       'Family_members', 'Title'])
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)





print(round(acc_sgd,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(round(acc_log,2,), "%")
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(round(acc_knn,2,), "%")
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)



Y_pred = gaussian.predict(X_test)



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print(round(acc_gaussian,2,), "%")
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print(round(acc_perceptron,2,), "%")
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(round(acc_linear_svc,2,), "%")
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



Y_pred = decision_tree.predict(X_test)



acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(round(acc_decision_tree,2,), "%")