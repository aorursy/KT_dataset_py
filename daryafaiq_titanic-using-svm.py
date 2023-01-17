from sklearn.model_selection import train_test_split

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.svm import SVC

from sklearn import svm

import numpy as np
titanic = pd.read_csv("../input/titanic/train.csv")

titanic.head()
titanic.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

titanic.head()
print("#Age missing entries =",titanic.Age.isnull().sum())

print("#survived missing entries =",titanic.Survived.isnull().sum())

print("#Pclass missing entries =",titanic.Pclass.isnull().sum())

print("#SibSp missing entries =",titanic.SibSp.isnull().sum())

print("#Parch missing entries =",titanic.Parch.isnull().sum())

print("#Fare missing entries =",titanic.Fare.isnull().sum())

print("#Cabin missing entries =",titanic.Cabin.isnull().sum())

print("#Embarked missing entries =",titanic.Embarked.isnull().sum())
titanic[titanic.Embarked.isnull()]
plt.subplots(figsize = (10,5))

sns.barplot(x = "Embarked", y = "Survived", data=titanic)

plt.title("P(Survived | Embarked)", fontsize = 25)

plt.xlabel("Embarked From", fontsize = 10);

plt.ylabel("Propability of Passenger Survived", fontsize = 10);
titanic.drop(['Cabin'],axis=1,inplace=True)

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic["Embarked"] = titanic["Embarked"].fillna("C")

titanic.Sex=titanic.Sex.replace(['male','female'],[0,1])

titanic.Embarked=titanic.Embarked.replace(['S','C','Q'],[0,1,2])

titanic.head()
print("#Age missing entries =",titanic.Age.isnull().sum())

print("#survived missing entries =",titanic.Survived.isnull().sum())

print("#Pclass missing entries =",titanic.Pclass.isnull().sum())

print("#SibSp missing entries =",titanic.SibSp.isnull().sum())

print("#Parch missing entries =",titanic.Parch.isnull().sum())

print("#Fare missing entries =",titanic.Fare.isnull().sum())

print("#Embarked missing entries =",titanic.Embarked.isnull().sum())
titanic_test = pd.read_csv("../input/titanic/test.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test.drop(['Cabin'],axis=1,inplace=True)

titanic_test.Sex=titanic_test.Sex.replace(['male','female'],[0,1])

titanic_test.Embarked=titanic_test.Embarked.replace(['S','C','Q'],[0,1,2])

titanic_test.head()
features = ["Pclass", "Sex", "Age", "SibSp", "Parch","Fare","Embarked"]
C = 0.7

clf = svm.SVC(kernel='linear',C=C).fit(titanic[features],titanic["Survived"])

prediction = clf.predict(titanic_test[features])
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],

                           "Survived":prediction})

submission.to_csv("titanic_result.csv", index=False)
submission.head()