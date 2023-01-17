import pandas as pd
import numpy as np
test_df=pd.read_csv("../input/titanic/test.csv")
train_df=pd.read_csv("../input/titanic/train.csv")
train_df.head(6)
test_df.head()
train_df.shape
test_df.shape
train_df.columns
test_df.columns
train_df["Survived"].value_counts()
train_df.info()
test_df.info()
train_df.describe()
test_df.describe()
train_df['Sex'].value_counts()
test_df['Sex'].value_counts()
train_df.describe(include=['O'])
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Pclass","Survived"]].groupby(['Pclass'],as_index=False).mean()
train_df[["Sex","Survived"]].groupby(['Sex'],as_index=False).mean()
test_df[["Pclass","Fare"]].groupby(['Pclass'],as_index=False).mean()  #as_index=false is necessary to represent index separate from pclass otherwise pclass represents as index numbers
train_df.isnull().sum()
train_df.Sex[train_df.Sex == 'male'] = 1
train_df.Sex[train_df.Sex == 'female'] = 2
train_df.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
train_df.Survived.value_counts().plot(kind="bar")  #This bar represents how many people are survived or non survived based on the no. 0 represents non survived and 1 represents survived
train_df.Sex.value_counts().plot(kind="bar",color=["r","g"])
train_df.plot(kind="scatter", x="Survived", y="Pclass")
train_df.plot(kind="scatter", x="Survived", y="Age")
train_df["Pclass"].value_counts().plot("bar")
train_df[train_df["Survived"] == 1]['Age'].value_counts().sort_index().plot("bar")
pd.crosstab(train_df.Pclass,train_df.Survived,margins=True).style.background_gradient(cmap="autumn_r")
train_df.head()
def Title(string):
    if string.find("Mr.")!=-1:
        return "0"
    elif string.find("Miss.")!=-1:
        return "1"
    elif string.find("Mrs.")!=-1 and string.find("Master.")==-1:
        return "2"
    else:
        return "3"
train_df["Title"] = train_df["Name"].apply(Title)
train_df.Title.value_counts()
test_df["Title"] = test_df["Name"].apply(Title)
test_df.Title.value_counts()
train_df = train_df.drop(["Name"],axis=1)
train_df.head(9)
train_df["Title"].value_counts().plot("bar")
train_df.head(9)
test_df.head()
test_df.drop(["Name"],axis=1,inplace=True)
test_df.head()
test_df["Sex"] = test_df["Sex"].map({"male": 1, "female": 2}).astype(int)
test_df
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
test_df
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df.head(15)
sns.boxplot(x='Survived',y='Fare',data=train_df)
train_df['family_size'] = train_df.SibSp + train_df.Parch
test_df['family_size'] = test_df.SibSp + test_df.Parch
test_df.head()
train_df.head()
train_df.Fare.value_counts()
train_df.head(9)
train_df["Embarked"] = train_df["Embarked"].fillna("S")
train_df.head(65)
train_df.Embarked=train_df.Embarked.map({"S":1, "C":2, "Q":3})
train_df.head()
test_df.Embarked=test_df.Embarked.map({"S":1, "C":2, "Q":3})
test_df.head()
test_df.head()
train_df.head()
def Age_bin(string):
    if string <= 15:
        return "0"
    elif string >15 and string <= 30:
        return "1"
    elif string >30 and string <=45:
        return "2"
    else:
        return "3"  # defining Age_bin for ranges of Age from 0 to 15, 15-30, 30-45 and 45- above.
train_df["Age_bin"] = train_df["Age"].apply(Age_bin)
train_df["Age_bin"].value_counts()
test_df["Age_bin"] = test_df["Age"].apply(Age_bin)
test_df["Age_bin"].value_counts()
train_df["Age_bin"].value_counts().plot(kind="pie")
train_df
test_df
train_df.drop(["PassengerId","SibSp","Parch","Ticket","Fare","Cabin","Age"],axis=1,inplace=True)
test_df.drop(["PassengerId","SibSp","Parch","Ticket","Fare","Cabin","Age"],axis=1,inplace=True)
train_df
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.copy()
X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()
logreg.fit(X_train, Y_train) # applied logic regression model by fitting it into X_train and Y_train data for prediction of test data set.
logreg.predict(X_test) # predicting the result of survived or not survived on test data.
logreg.score(X_train,Y_train)