import pandas as pd 

import numpy as np
df=pd.read_csv(r'../input/titanic/train.csv')
df.head(2)
df=df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
df.head(2)
dummies=pd.get_dummies(df.Sex)
dummies.head(2)
df1=pd.concat([df,dummies],axis='columns')
df1.head(2)
df1.isnull().sum()
df1=df1.dropna()
df1.isnull().sum()
df1['Age'].describe()
lower_hinge=df1.Age.quantile(0.25)

lower_hinge
upper_hinge=df1.Age.quantile(0.75)

upper_hinge
IQR=upper_hinge-lower_hinge

IQR
floor=lower_hinge-1.5*IQR

floor
cap=upper_hinge+1.5*IQR

cap
df1=df1[(df1['Age']>floor)&(df1['Age']<cap)]

df1.Age.describe()
x=df1.drop(['Survived','Sex'],axis='columns')
x.head(2)
y=df1.Survived
y.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
model1=LogisticRegression()
model1.fit(x,y)
test=pd.read_csv(r'../input/titanic/test.csv')
test.head()
test=test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
test.head(2)
test_dummies=pd.get_dummies(df.Sex)
test_dummies.head(2)
test_df1=pd.concat([df,dummies],axis='columns')
test_df1.isnull().sum()
test_df1=test_df1.dropna()
test_df1.isnull().sum()
test_x=test_df1.drop(['Survived','Sex'],axis='columns')
test_x.head(2)
test_y=test_df1.Survived
test_y.head(2)
model1.score(test_x,test_y)
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB
score_s=cross_val_score(SVC(kernel='rbf'),test_x,test_y)

np.average(score_s)
score_d=cross_val_score(DecisionTreeClassifier(),test_x,test_y)

np.average(score_d)
score_r=cross_val_score(RandomForestClassifier(n_estimators=10),test_x,test_y)

np.average(score_r)
score_g=cross_val_score(GaussianNB(),test_x,test_y)

np.average(score_g)
score_lr=cross_val_score(LogisticRegression(),test_x,test_y)

np.average(score_lr)
score_m=cross_val_score(MultinomialNB(),test_x,test_y)

np.average(score_m)