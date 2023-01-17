from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt

print("imported!")
Data=pd.read_csv('../input/train.csv')
Data.head()


Data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

Data.head()


print("#Age missing entries =",Data.Age.isnull().sum())

print("#survived missing entries =",Data.Survived.isnull().sum())

print("#Pclass missing entries =",Data.Pclass.isnull().sum())

print("#SibSp missing entries =",Data.SibSp.isnull().sum())

print("#Parch missing entries =",Data.Parch.isnull().sum())

print("#Fare missing entries =",Data.Fare.isnull().sum())

print("#Cabin missing entries =",Data.Cabin.isnull().sum())

print("#Embarked missing entries =",Data.Embarked.isnull().sum())







Data[Data.Embarked.isnull()]


g = sns.catplot(x="Embarked", y="Survived",  data=Data,

                   height=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")



Data=Data.fillna({'Embarked':'C'})



print("#Embarked missing entries =",Data.Embarked.isnull().sum())
Data=Data.fillna({"Cabin":'X'})

Data.head()
Data["Cabin"]=Data["Cabin"].str.slice(0,1)
Data.head(10)
plot1=sns.catplot(x="Cabin", y="Survived",  data=Data,

                   height=6, kind="bar", palette="muted")
Data['Cabin']=Data['Cabin'].replace(['A','B','C','D','E','F','G','T','X'],[0,1,2,3,4,5,6,7,8])

Data.head(10)
# Converting other categorical features as well.

Data['Sex']=Data['Sex'].replace(['male','female'],[0,1])

Data['Embarked']=Data['Embarked'].replace(['S','C','Q'],[0,1,2])

Data.head()

sns.heatmap(Data[["Age","Sex","SibSp","Parch","Pclass","Embarked","Fare","Cabin","Survived"]].corr(),annot=True)
age_means=np.zeros((3,9))

median=Data.Age.mean()

for classNum in range (0,Data.Pclass.max()):  # 0 --> 1st class

    for sibNum in range (0,Data.SibSp.max()+1): # adding one to take the range [0,8] not [0,8[.

        age_means[classNum][sibNum]=Data["Age"][(Data["Pclass"]==(classNum+1)) & (Data["SibSp"]==sibNum)].mean()

        if np.isnan(age_means[classNum][sibNum]):

            age_means[classNum][sibNum]=median
print(age_means)
Null_indx=list(Data["Age"][Data["Age"].isnull()].index)
for i in Null_indx:

    Data["Age"].iloc[i]=age_means[Data.Pclass[i] - 1][Data.SibSp[i]]

    

print("#Age missing entries =",Data.Age.isnull().sum())

Y=Data.Survived

X=Data

X.drop(['Survived'],axis=1,inplace=True)

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2, random_state=3)

classifier= GaussianNB()

classifier.fit(X_train, Y_train)

classifier.class_prior_

predicts=classifier.predict(X_test)

accuracy=round(accuracy_score(predicts,Y_test),3)

print(accuracy)
test=pd.read_csv('../input/test.csv')

IDs=test.PassengerId

test.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

test.head()
test.Sex=test.Sex.replace(['male','female'],[0,1])

test.Embarked=test.Embarked.replace(['S','C','Q'],[0,1,2])

test=test.fillna({"Cabin":'X'})

test["Cabin"]=test["Cabin"].str.slice(0,1)

test.head()
Null_test=list(test["Age"][test["Age"].isnull()].index)

for i in Null_test:

    test["Age"].iloc[i]=age_means[test.Pclass[i] - 1][test.SibSp[i]]

print("#Age missing entries =",Data.Age.isnull().sum())

print("#Pclass missing entries =",test.Pclass.isnull().sum())

print("#SibSp missing entries =",test.SibSp.isnull().sum())

print("#Parch missing entries =",test.Parch.isnull().sum())

print("#Fare missing entries =",test.Fare.isnull().sum())

print("#Embarked missing entries =",test.Embarked.isnull().sum())

test['Cabin']=test['Cabin'].replace(['A','B','C','D','E','F','G','T','X'],[0,1,2,3,4,5,6,7,8])

test=test.fillna({'Fare':34})

subPredictions=classifier.predict(test)

subFile=pd.DataFrame({'PassengerId': [],'Survived':[]})

subFile.PassengerId=IDs

subFile.Survived=subPredictions

subFile.to_csv( 'MySubmissionCabin' ,index=False)

subFile.head()