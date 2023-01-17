

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv('../input/credit-card/credit card taiwan svm algorithm.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
#Null Value Handle

m=df['AGE'].mean()
m
df.AGE=df.AGE.fillna(m)
df.isnull().sum()
##Split Features and Label

x=df.drop(['default.payment.next.month'],axis=1)
y=df['default.payment.next.month']
y
##Train and test size
from sklearn.model_selection import train_test_split


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20,random_state=1)
xtest
#fit the model

from sklearn.svm import SVC
sv=SVC(gamma=True)
sv
sv.fit(xtrain,ytrain)
##Accuracy 
sv.score(xtest,ytest)
sv.predict(xtest)
xtest
ytest
##Logistic Regresion

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)
##Decission Tree Classifier

from sklearn.tree import  DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dt.score(xtest,ytest)
##Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
rf.score(xtest,ytest)
## So Random Forest is good here and work fastly bcz of Big Datas
from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(xtrain,ytrain)
gb.score(xtest,ytest)
from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(xtrain,ytrain)
bn.score(xtest,ytest)
from sklearn.naive_bayes import MultinomialNB
mn=MultinomialNB()
mn.fit(xtrain,ytrain)
