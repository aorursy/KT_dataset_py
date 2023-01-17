import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(5)
test.head(5)
train.describe()
test.describe()
train["Title"] = train["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip().upper())
test["Title"] = test["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip().upper())
train.head(5)
test.head(5)
train.Title.value_counts()
train[train.Age.notnull()].query("Title =='MASTER' and Age >= 10")
train[train.Age.notnull()].query("Title =='MASTER' and Age >= 10").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MISS' and Age <= 16").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MISS' and Age > 16 and Age < 30").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MISS' and Age >= 30 and Age < 55").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MISS' and Age > 55").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MR' and Age < 16").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MR' and Age >= 16 and Age < 30").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MR' and Age >= 30 and Age < 55").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MRS' and Age < 16").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MRS' and Age >= 16 and Age < 30").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MRS' and Age >= 30 and Age < 55").Survived.value_counts()
train[train.Age.notnull()].query("Title =='MRS' and Age >=55").Survived.value_counts()
train[train.Age.notnull()].query( "Age >=55").Survived.value_counts()
train[train.Age.isnull()].Title.value_counts()
def AgeGroup(title, age):
    if pd.isnull(age):
        if (title == "MASTER"):
            return "KID"
        elif (title == "MR" or title =="MISS"):
            return "YOUNGADULT"
        else:
            return "ADULT"
    elif (age < 16):
        return "KID"
    elif (age >= 16 and age < 30 ):
        return "YOUNGADULT"
    elif (age>=30 and age <=55):
        return "ADULT"
    else:
        return "OLD"
train["AgeGroup"] = np.vectorize(AgeGroup)(train["Title"],train["Age"])
test["AgeGroup"] = np.vectorize(AgeGroup)(test["Title"],test["Age"])
train.head(5)
test.head(5)
test=pd.concat([test,pd.get_dummies(test.Sex)],axis=1)
test=pd.concat([test,pd.get_dummies(test.AgeGroup)],axis=1)
train=pd.concat([train,pd.get_dummies(train.Sex)],axis=1)
train=pd.concat([train,pd.get_dummies(train.AgeGroup)],axis=1)
test.head(5)
train.head(5)
trainingSetSurvivedPassengerVector =train.Survived
train.columns
train.drop(['Survived','Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup'], axis=1, inplace=True)
train.head(5)
test.drop(['Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup'], axis=1, inplace=True)
test.head(5)
regression = LogisticRegression()
regression.fit(train, trainingSetSurvivedPassengerVector)
testSetSurvivedPassengerVector = regression.predict(test)
modelAccuracy = round(regression.score(train, trainingSetSurvivedPassengerVector) * 100, 2)
modelAccuracy
submitResult = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": testSetSurvivedPassengerVector})
submitResult.to_csv('submitResults.csv', index=False)
