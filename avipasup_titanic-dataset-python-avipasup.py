import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

#trainurl="https://raw.githubusercontent.com/mdelhey/kaggle-titanic/master/Data/train.csv"

#testurl="https://raw.githubusercontent.com/mdelhey/kaggle-titanic/master/Data/test.csv"

titrain=pd.read_csv("/kaggle/input/train.csv", sep=",")

titest=pd.read_csv("/kaggle/input/test.csv", sep=",")
#titrain.head()

#(pd.isnull(titrain)).sum()

titrain.info()
titrain.describe(include="all").transpose()
titrain['Age'] = titrain.groupby(['Survived','Parch','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))



titrain['CabinID']=titrain['Cabin'].str[0]

cabingrp=titrain.groupby(['CabinID','Pclass'])['Fare'].mean()
titrain=titrain.drop(['Cabin','CabinID'], axis=1)

titrain.info()

titrain[titrain['Embarked'].isnull()]


#titrain[titrain['Ticket'] > 80.0][:]

#titrain[titrain['Ticket'].str.len()==5][:]



titrain[titrain['Ticket'].str[0:4]=='1135'][:]
titrain.groupby('Embarked')['PassengerId'].nunique()
titrain['Embarked']=titrain['Embarked'].transform(lambda x: x.fillna("S"))

titrain.info()
#Creating an family size feature from the existing SibSp and Parch features by adding both features

titrain['familysize']=titrain['SibSp']+titrain['Parch']





###Normalizing Data



#dropping unwanted features

titrainlog=titrain.drop(['PassengerId','Name','Ticket'], axis=1)

# converting sex to numeric variable

titrainlog['Sex']=titrainlog.Sex.map({'male':1,'female':0}) 

titrainlog.head()
titraindummy=pd.get_dummies(titrainlog, columns=['Pclass', 'Embarked'])

#titraindummy['Intercept']=1

titraindummy.info()
traindata=titraindummy.drop(['Survived','Embarked_S','Pclass_2'], axis=1)

traintarget=titraindummy.drop(['Sex','Age','SibSp','Parch','Fare','familysize','Pclass_1',

'Pclass_3','Embarked_C','Embarked_Q','Embarked_S','Pclass_2'], axis=1)

#list(traindata)

#converting y (target variable) to a 1D array

y=np.ravel(traintarget)

logistic=LogisticRegression()

model=logistic.fit(traindata,y)
model.coef_
traindata=sm.add_constant(traindata)

modelogit=sm.GLM(traintarget,traindata, 

                 family = sm.families.Binomial(sm.families.links.logit))

#sm.families.Binomial(sm.Logit).fit()



#const is the intercept

modelogit.fit().summary()