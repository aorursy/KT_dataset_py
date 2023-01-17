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
import matplotlib.pyplot as plt

%matplotlib inline

from statistics import mode

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import metrics

import math

import re

import seaborn as sns
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data

test_data
train_data.info()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
new_age_train = pd.unique(train_data["Age"])

mean_age_train = round(np.nanmean(new_age_train),0)

print("Average Age is ", mean_age_train)

                       
train_data["Age"]=train_data["Age"].fillna(34)
train_data.info()
new_age_test = pd.unique(test_data["Age"])

mean_age_test = round(np.nanmean(new_age_test),0)

print("Average Age is ", mean_age_test)
test_data["Age"]=test_data["Age"].fillna(31)
test_data.info()
plt.figure(figsize=(3,4))

sns.set(style="darkgrid")

sns.countplot(x="Survived", data=train_data,palette=["Red","Blue"])

plt.title("Survived",size=20)

plt.show()
rate_survived = round(np.mean(train_data["Survived"]),3)*100

print("Survival Rate",rate_survived,"%")
plt.figure(figsize=(3,4))

sns.set(style="darkgrid")

sns.countplot(x="Sex", data=train_data,palette=["Red","Blue"])

plt.title("Survived male or female",size=20)

plt.show()
total= train_data["Survived"].sum()

print(total)
men= train_data[train_data["Sex"]=="male"]["Sex"].count()

print(men)
women= train_data[train_data["Sex"]=="female"]["Sex"].count()

print(women)
plt.figure(figsize=(5,5))

sns.countplot(x="Survived", hue="Sex",data=train_data)

plt.title("Survived relation with gender",size=20)
plt.figure(figsize=(5,5))

sns.countplot(x="Survived",hue="Pclass",data=train_data)

plt.title("Relation : Survived and Pclass",size=20)
plt.figure(figsize=(5,5))

sns.countplot(x="Survived",hue="SibSp",data=train_data)

plt.title("Relation : Survived and SibSp",size=20)
plt.figure(figsize=(5,5))

sns.countplot(x="Survived",hue="Parch",data=train_data)

plt.title("Relation : Survived and Parch",size=20)
plt.figure(figsize=(5,5))

sns.countplot(x="Survived",hue="Embarked",data=train_data)

plt.title("Relation : Survived and Embarked",size=20)
sns.heatmap(train_data.corr(),annot=True)
print(train_data.isnull().sum())
si = SimpleImputer(strategy="most_frequent")

train_data["Embarked"]=si.fit_transform(train_data["Embarked"].values.reshape(-1,1))

test_data["Embarked"]=si.fit_transform(test_data["Embarked"].values.reshape(-1,1))

print(train_data.isnull().sum())

train_data.head()
train_data["Embarked"][train_data["Embarked"]=="S"]=0

train_data["Embarked"][train_data["Embarked"]=="C"]=1

train_data["Embarked"][train_data["Embarked"]=="Q"]=2



test_data["Embarked"][test_data["Embarked"]=="S"]=0

test_data["Embarked"][test_data["Embarked"]=="C"]=1

test_data["Embarked"][test_data["Embarked"]=="Q"]=2



train_data["Sex"][train_data["Sex"]=="male"]=0

train_data["Sex"][train_data["Sex"]=="female"]=1



test_data["Sex"][test_data["Sex"]=="male"]=0

test_data["Sex"][test_data["Sex"]=="female"]=1



train_data.head()
si_fare = SimpleImputer(missing_values=np.nan, strategy="median")

test_data["Fare"]=si_fare.fit_transform(test_data["Fare"].values.reshape(-1,1))

test_data.info()
mmx=MinMaxScaler()

train_data["Fare"]=mmx.fit_transform(train_data["Fare"].values.reshape(-1,1))

test_data["Fare"]=mmx.fit_transform(test_data["Fare"].values.reshape(-1,1))



train_data["Age"]=mmx.fit_transform(train_data["Age"].values.reshape(-1,1))

test_data["Age"]=mmx.fit_transform(test_data["Age"].values.reshape(-1,1))



train_data
train_data["Title"]= train_data["Name"].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())

test_data["Title"]= test_data["Name"].map(lambda x: re.compile("([A-Za-z]+)\.").search(x).group())

print(train_data["Title"].unique())
print(test_data["Title"].unique())
title_mapping = {'Mr.': 0, 'Mrs.': 0, 'Miss.': 0, 'Master.' : 1,'Don.': 1, 'Rev.' : 1,'Dr.' : 1,'Mme.': 0, 'Ms.': 0, 'Major.': 1,

 'Lady.': 1, 'Sir.': 1, 'Mlle.': 0, 'Col.': 1, 'Capt.': 1, 'Countess.': 1, 'Jonkheer.': 1,'Dona.': 1,}



train_data['Title'] = train_data['Title'].map(title_mapping)

train_data['Title'] = train_data['Title'].fillna(0)



print(train_data['Title'].unique())
title_mapping = {'Mr.': 0, 'Mrs.': 0, 'Miss.': 0, 'Master.' : 1,'Don.': 1, 'Rev.' : 1,'Dr.' : 1,'Mme.': 0, 'Ms.': 0, 'Major.': 1,

 'Lady.': 1, 'Sir.': 1, 'Mlle.': 0, 'Col.': 1, 'Capt.': 1, 'Countess.': 1, 'Jonkheer.': 1,'Dona.': 1,}



test_data['Title'] = test_data['Title'].map(title_mapping)

test_data['Title'] = test_data['Title'].fillna(0)



print(test_data['Title'].unique())
for n, i in enumerate(train_data["SibSp"]):

    if i != 0:

     train_data["SibSp"][n] = 1





for m, k in enumerate(train_data["Parch"]):

    if k != 0:

     train_data["Parch"][m] = 1
DS_train = train_data.drop(columns=["Cabin","Name","Ticket"])

DS_train
DS_test = test_data.drop(columns=["Cabin","Name","Ticket"])

DS_test


DS_train = DS_train[['PassengerId','Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title','Survived']]

DS_train


DS_test = DS_test[['PassengerId','Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title']]

DS_test


feature_columns = ['Sex','Pclass','Age','Fare','SibSp','Parch','Embarked','Title']

X = DS_train[feature_columns] 

y = DS_train['Survived']     

X
X_train,X_val,Y_train,Y_val=train_test_split(X,y,test_size=0.30,random_state=0)

logreg=LogisticRegression(max_iter=30000)

logreg.fit(X_train,Y_train)

y_pred=logreg.predict(X_val)
acc_log = round(accuracy_score(y_pred,Y_val)*100,2)
print(acc_log)
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

svc = SVC()

svc.fit(X_train,Y_train)

svc_pred = svc.predict(X_val)

acc_svc=round(accuracy_score(svc_pred,Y_val)*100,2)
print(acc_svc)
randomforest = RandomForestClassifier()

randomforest.fit(X_train, Y_train)

y_pred = randomforest.predict(X_val)

acc_randomforest = round(accuracy_score(y_pred, Y_val) * 100, 2)

print(acc_randomforest)


gbc = GradientBoostingClassifier()

gbc.fit(X_train, Y_train)

y_pred = gbc.predict(X_val)

acc_gbc = round(accuracy_score(y_pred, Y_val) * 100, 2)

print(acc_gbc)
from catboost import CatBoostClassifier, cv, Pool



clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=2)

clf.fit(X_train,Y_train, eval_set=(X_val,Y_val), early_stopping_rounds=100,verbose=False)



y_pred = clf.predict(X_val)

acc_clf = round(accuracy_score(y_pred, Y_val) * 100, 2)

print(acc_clf)
ids = DS_test['PassengerId']

predictions = gbc.predict(DS_test.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)