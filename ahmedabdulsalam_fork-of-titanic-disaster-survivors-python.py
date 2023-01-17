#Lets import Packages and Read data..



#analytics packages

import pandas as pd

import numpy as np



#visualization packages

import matplotlib.pyplot as plt



#reading data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Lets take a look at our data.

train.head()
#what are the values that have 'NULL'?

pd.isnull(train).sum()
pd.isnull(test).sum()
test_ID = test.PassengerId #used in End_game

train = train.drop(['PassengerId','Name','Ticket'],axis=1)

test = test.drop(['Name','Ticket'],axis=1)

#axis = 1 is nothing but drop ROW wise.. 
#Filling the null values of 'Age'!

age_mean_train=train['Age'].mean()

train['Age']=train['Age'].fillna(age_mean_train)

####

age_mean_test=test['Age'].mean()

test['Age']=test['Age'].fillna(age_mean_test)

#Time to fill the null values of 'Embarked'

#for that we can't simply fill anything..so lets see where majority of people embarked from..

print("From train data")

southampton = train[train["Embarked"] == "S"].shape[0]

cherbourg = train[train["Embarked"] == "C"].shape[0]

queenstown = train[train["Embarked"] == "Q"].shape[0]



print("No. of people from Southampton (S) = ",southampton)

print("No. of people from Cherbourg   (C) = ",cherbourg)

print("No. of people from Queenstown  (Q) = ",queenstown)

####

print("\nFrom test data")

southampton = train[train["Embarked"] == "S"].shape[0]

cherbourg = train[train["Embarked"] == "C"].shape[0]

queenstown = train[train["Embarked"] == "Q"].shape[0]



print("No. of people from Southampton (S) = ",southampton)

print("No. of people from Cherbourg   (C) = ",cherbourg)

print("No. of people from Queenstown  (Q) = ",queenstown)
#now that we see majority of people are from Southampton.. so we replace embarked with (S)..

train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")
#time to fill null values in 'Fare'..



fare_median=train["Fare"].median()

train["Fare"] = train["Fare"].fillna(fare_median)

####

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
#cabin is full of null values.. thus impacting on our prediction.. 

#so lets simply remove null values

train.drop("Cabin",axis=1,inplace=True)

test.drop("Cabin",axis=1,inplace=True)
#lets see if we have any null values

pd.isnull(train).sum()
pd.isnull(test).sum()
#lets see which variables have what datatype

train.dtypes
test.dtypes
#lets use Quartiles to create Age groups..

train.Age.describe()
#divide based on quartiles

AgeTypeTrain = []

for row in train["Age"]:

    if row >= 48:

        AgeTypeTrain.append("1")

    elif row >=39:

        AgeTypeTrain.append("2")

    elif row >= 28:

        AgeTypeTrain.append("3")

    else:

        AgeTypeTrain.append("4")

train["Age"]=AgeTypeTrain
#now for test data

test.Age.describe()
#divide based on quartiles

AgeTypeTest = []

for row in test["Age"]:

    if row >= 35:

        AgeTypeTest.append("1")

    elif row >=30:

        AgeTypeTest.append("2")

    elif row >= 23:

        AgeTypeTest.append("3")

    else:

        AgeTypeTest.append("4")

test["Age"]=AgeTypeTest
train["Age"].head()
test["Age"].head()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
#we convert the variables into numeric using lable Encoder..

train["Sex"]=le.fit_transform(train["Sex"].values)

train["Embarked"]=le.fit_transform(train["Embarked"].values)



####

test["Sex"]=le.fit_transform(test["Sex"].values)

test["Embarked"]=le.fit_transform(test["Embarked"].values)
#now lets see our Data Frames

train.head()
test.head()
from sklearn.linear_model import LogisticRegression

glm=LogisticRegression()
# define training and testing sets

X_train = train.drop("Survived",axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId",axis=1).copy()
#fit the data..

glm.fit(X_train,Y_train)
#now lets predict!

predicted = glm.predict(X_test)

#time to know how accurate our model is

print("Accurcy = %.2f" %round(glm.score(X_train, Y_train) * 100, 2))
#lets export our predicted output.

submission = pd.DataFrame({

        "PassengerId":test_ID,

        "Survived": predicted

    })

submission.head()

submission.to_csv('titanic_result_pandas.csv', index=False)