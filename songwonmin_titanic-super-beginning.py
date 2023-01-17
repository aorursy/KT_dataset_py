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

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test.describe(include="all")
train
gender_submission
train.columns
train.sample(10)
len(test)
train.Sex
train["Sex"]
sns.barplot(x="Sex", y="Survived",data=train)
train.columns
sns.barplot(x="Pclass",y="Survived",data=train)
sns.barplot(x="Name",y="Survived",data=train)
sns.barplot(x="Age",y="Survived",data=train)
sns.barplot(x="SibSp",y="Survived",data=train)
sns.barplot(x="Parch",y="Survived",data=train)
sns.barplot(x="Ticket",y="Survived",data=train)
sns.barplot(x="Fare",y="Survived",data=train)
sns.barplot(x="Cabin",y="Survived",data=train)
sns.barplot(x="Embarked",y="Survived",data=train)
train["Age"].isnull().sum()
train["Age"].hist()
train["Age"]= train["Age"].fillna(-0.5)

test["Age"]= test["Age"].fillna(-0.5)

bins = [-1,0,5,12,18,24,35,60,np.inf]

labels=["Unknown","Baby","Child","Teenager","Student","Young Adult","Adult","Senior"]

train["AgeGroup"]=pd.cut(train["Age"],bins,labels=labels)

test["AgeGroup"]=pd.cut(train["Age"],bins,labels=labels)

sns.barplot(x="AgeGroup",y="Survived",data=train)

plt.show
train["Cabin"].isnull().sum()
train["CabinBool"]=train["Cabin"].notnull().astype("int")

test["CabinBool"]=test["Cabin"].notnull().astype("int")

sns.barplot(x="CabinBool",y="Survived",data=train)
test.describe(include="all")
train=train.drop(["Cabin"],axis=1)

test=test.drop(["Cabin"],axis=1)
train=train.drop(["Ticket"],axis=1)

test=test.drop(["Ticket"],axis=1)
sns.barplot(x="Embarked",y="Survived",data=train)
train.Embarked.hist()
train=train.fillna({"Embarked":"S"})
train=train.drop(["Name"],axis=1)

test=test.drop(["Name"],axis=1)
age_group_mapping={1:"Baby",2:"Child",3:"Teenager",4:"Student",5:"Young Adult",6:"Adult",7:"Senior"}
train["AgeGroup"]=train["AgeGroup"].map(age_group_mapping)

test["AgeGroup"]=test["AgeGroup"].map(age_group_mapping)
train.isnull().sum()
train["AgeGroup"]=train["AgeGroup"].fillna("0")

test["AgeGroup"]=test["AgeGroup"].fillna("0")
train["AgeGroup"]
train
train=train.drop(["Age"],axis=1)

test=test.drop(["Age"],axis=1)
sex_group_mapping={"male":0,"female":1}

train["Sex"]=train["Sex"].map(sex_group_mapping)

test["Sex"]=test["Sex"].map(sex_group_mapping)
embarked_group_mapping={"S":1,"C":2,"Q":3}

train["Embarked"]=train["Embarked"].map(embarked_group_mapping)

test["Embarked"]=test["Embarked"].map(embarked_group_mapping)
test.Fare.mean()
test["Fare"]=test.Fare.fillna(test.Fare.mean())
from sklearn.model_selection import train_test_split

predictors = train.drop(["Survived","PassengerId"], axis=1)

target = train["Survived"]

x_train,x_val,y_train,y_val = train_test_split(predictors,target,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred=gaussian.predict(x_val)

acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_gaussian)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred=logreg.predict(x_val)

acc_logreg=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_logreg)
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred=perceptron.predict(x_val)

acc_perceptron=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_perceptron)
from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred=svc.predict(x_val)

acc_svc=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_svc)
from sklearn.ensemble import RandomForestClassifier



rforest = RandomForestClassifier()

rforest.fit(x_train, y_train)

y_pred=rforest.predict(x_val)

acc_rforest=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_rforest)
ids = test["PassengerId"]

preds = rforest.predict(test.drop('PassengerId', axis=1))
output = pd.DataFrame({"PassengerId":ids, "Survived":preds})

output.to_csv("submission.csv", index=False)