import pandas as pd
import numpy as np

from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")


#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



print("Hello World")
titanic_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

 
titanic_df.head()

titanic_df.info()
print("::::::::::::::::::::::::::::::::::::")
test_df.info()
drop_columns = ["PassengerId","Name","Ticket"]

drop_columns_test = ["Name","Ticket"]

titanic_df.drop(drop_columns, axis =1, inplace = True)
test_df.drop(drop_columns_test , axis =1 , inplace =True)

print(titanic_df["Embarked"].isnull().sum())
print(test_df["Embarked"].isnull().sum())

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

print(titanic_df["Embarked"].isnull().sum())
titanic_df.head()
sns.factorplot("Embarked", "Survived", data = titanic_df, aspect =2 )
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x="Survived", data=titanic_df, ax=axis1)
sns.countplot(x="Survived", hue = "Embarked", data= titanic_df , ax = axis2)

embark_perc = titanic_df[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean()

sns.barplot(x='Embarked', y ="Survived", data= embark_perc,order=['S', 'C', 'Q'], ax= axis3)


titanic_dummy= pd.get_dummies(titanic_df["Embarked"])
test_dummy = pd.get_dummies(test_df["Embarked"])

test_dummy.head()
titanic_df = titanic_df.join(titanic_dummy)
test_df = test_df.join(test_dummy)

titanic_df.drop(["Embarked"], axis = 1, inplace = True)
#Fare

print(test_df["Fare"].isnull().sum())

test_df["Fare"].fillna(test_df["Fare"].median(), inplace= True)

titanic_df["Fare"].astype(int)
test_df["Fare"].astype(int)
titanic_df["Age"].fillna(0).astype(int)
test_df["Age"].fillna(0).astype(int)

titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)




titanic_df["Survived"].astype(int)

print(titanic_df["Age"].isnull().sum())
print(titanic_df["Age"][titanic_df["Age"].isnull()])
print(test_df["Age"].median())
Y_train  = titanic_df["Survived"] 
get_train_dummies_gender = pd.get_dummies(titanic_df["Sex"])
get_test_dummies_gender = pd.get_dummies(test_df["Sex"])
titanic_df=titanic_df.join(get_train_dummies_gender)
test_df=test_df.join(get_test_dummies_gender)
X_train  = titanic_df.drop(["Survived","Cabin","Sex"], axis =1)
X_test = test_df.drop(["PassengerId","Cabin","Sex","Embarked"], axis = 1)
print(Y_train.head())
print(X_train.head())
print(X_test.head())
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

