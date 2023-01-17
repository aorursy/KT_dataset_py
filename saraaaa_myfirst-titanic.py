#imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#load our data

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train
y_train=pd.DataFrame(train["Survived"])

x_train=train.drop(["PassengerId","Name","Ticket","Cabin","Fare"],1)

x_test=test.drop(["PassengerId","Name","Ticket","Cabin","Fare"],1)

y_test=gender_submission.drop(["PassengerId"],1)
x_train
y_train
x_test
y_test
x_train.info()
x_test.info()
y_train.info()
y_test.info()
print(x_train["Age"].isna().sum())

print(x_test["Age"].isna().sum())

print(x_train["Embarked"].isna().sum())
x_train["Age"]=x_train["Age"].fillna(value=x_train["Age"].mean())

x_test["Age"]=x_test["Age"].fillna(value=x_test["Age"].mean())

x_train["Embarked"]=x_train["Embarked"].fillna(value=x_train["Embarked"].mode()[0])

x_test["Embarked"]=x_test["Embarked"].fillna(value=x_test["Embarked"].mode()[0])
print(x_train["Age"].isna().sum())

print(x_test["Age"].isna().sum())

print(x_train["Embarked"].isna().sum())
x_train
x_test
x_train.Embarked.unique()
sex={'male':0,'female':1}

emb={'Q':0,'S':1,'C':2}
x_train["Sex"]=x_train["Sex"].map(sex)

x_train["Embarked"]=x_train["Embarked"].map(emb)

x_test["Sex"]=x_test["Sex"].map(sex)

x_test["Embarked"]=x_test["Embarked"].map(emb)
x_train
x_test
y_train.hist(column='Survived')
x_train.hist(column='Pclass')
x_train.hist(column='Sex')
x_train.hist(column='Embarked')
x_train.hist(column='Parch')
x_train.hist(column='SibSp')
x_train.hist(column='Age',bins=20)
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Pclass"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Pclass"],kde_kws={"label":"died"})
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Age"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Age"],kde_kws={"label":"died"})
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Sex"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Sex"],kde_kws={"label":"died"})
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["SibSp"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["SibSp"],kde_kws={"label":"died"})
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Parch"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Parch"],kde_kws={"label":"died"})
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Embarked"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Embarked"],kde_kws={"label":"died"})
pd.cut(x_train["Age"],bins=4).unique()
def factorize_age(col):

    for i,e in enumerate(col):

        if col[i] > 0.34 and col[i] <=20.315:

            col[i]=0

        elif col[i] > 20.315 and col[i] <=40.21:

             col[i]=1

        elif col[i] >40.21 and col[i] <=60.105:

            col[i]=2

        elif col[i] > 60.105 and col[i] <=80.0:

            col[i]=3

    return col
factorize_age(x_train["Age"])

factorize_age(x_test["Age"])


x_train
x_test
fig=plt.figure(figsize=(10,10))

sns.distplot(x_train.loc[x_train["Survived"]==1]["Age"],kde_kws={"label":"Survived"})

sns.distplot(x_train.loc[x_train["Survived"]==0]["Age"],kde_kws={"label":"died"})
x_train=x_train.drop(["Survived"],1)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
x_train
from sklearn.linear_model import LogisticRegression

regressor1=LogisticRegression(random_state=0)
regressor1.fit(x_train,y_train)
y_pred1=regressor1.predict(x_test)
from sklearn.metrics import confusion_matrix

cm1=confusion_matrix(y_test,y_pred1)
cm1
print((255+144)/(266+152))
from sklearn.svm import SVC

regressor2=SVC(kernel='rbf',random_state=0)

regressor2.fit(x_train,y_train)
y_pred2=regressor2.predict(x_test)

cm2=confusion_matrix(y_test,y_pred2)
cm2
print((254+123)/(254+123+12+29))
gender_submission
y_pred1=pd.DataFrame(y_pred1)

y_pred1
y_pred1["PassengerId"]=gender_submission["PassengerId"]

y_pred1["Survived"]=y_pred1.loc[:,0]

y_pred1=y_pred1.drop(y_pred1.columns[0],1)
y_pred1
y_pred1.to_csv('SurvivedPrediction.csv',index=False)