#importing base libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#magic function to view plot inside jupyter notebook
%matplotlib inline 

#loading datasets from Kaggle
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")

import missingno as msno #for visualizing missing data
#train dataset missing data and preview
print("Train data rows x columns: ",train.shape,"\n")
print(train.isnull().sum())
train.head()
#test dataset missing data and preview
print("Test data rows x columns: ",train.shape,"\n")
print(test.isnull().sum())
test.head()
train.head()
#visualize missing data in train as matrix
msno.matrix(train)
#visualize missing data in train as matrix
msno.matrix(test)
#visualize missing data in train as bar chart
msno.bar(train)
#visualize missing data in test as bar chart
msno.bar(test)
#check if more than 40% of information is missing in columns for train data
print(train.isnull().sum()>int(0.40*train.shape[0]))
#check if more than 40% of information is missing in columns for test data
print(train.isnull().sum()>int(0.40*train.shape[0]))
#Data is mssing in Age columns. Histogram plot for train data.
sns.distplot(train['Age'].dropna(),hist=True, kde=True,rug=True, bins=40)
#Histogram plot for test data
sns.distplot(test['Age'].dropna(),hist=True, kde=True, bins=40, rug=True)
sns.countplot(x="Survived",data=train,palette="deep")
sns.countplot(x="Survived",hue="Pclass",data=train,palette="deep")
corr =train.corr()
sns.heatmap(corr,annot=True)
#Check the data type of each feature
train.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
#intantiate label encoder
le=LabelEncoder()
#Combining both datasets
dataset=[train,test]
#Dropping the PassengerId column only in train as we equire passengerid in our test dataset for submission
train.drop("PassengerId",axis=1,inplace=True)
train["Name"]
for data in dataset:
    data["Title"]=data["Name"].str.extract('([A-Za-z]+)\.',expand=False)
train["Title"].value_counts()
test["Title"].value_counts()
#create mapping
title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Col":3,"Rev":3,"Ms":3,"Dr":3,"Dona":3,"Major":3,
         "Mlle":3,"Countess":3,"Sir":3,"Jonkheer":3,"Don":3,"Mme":3,"Capt":3,"Lady":3}
for data in dataset:
    data["Title"]=data["Title"].map(title_mapping)
#check the count of survived according ot titles
sns.countplot(x="Survived",hue="Title",data=train,palette="deep")
#Dropping the column Names as it is no longer required
for data in dataset:
    data.drop("Name",axis=1,inplace=True)
for data in dataset:
    data["Sex"]=le.fit_transform(data["Sex"])
Skewness = 3*(train["Age"].mean()-train["Age"].median())/train["Age"].std()
Skewness
for data in dataset:
    data["Age"].fillna(data.groupby("Title")["Age"].transform("median"),inplace=True)
for data in dataset:
    data["Family Size"]=data["SibSp"]+data["Parch"]+1

#dropping the Parch and SibSp columns
for data in dataset:
    data.drop(["SibSp","Parch"],axis=1,inplace=True)
#dropping Ticket column
for data in dataset:
    data.drop(["Ticket"],axis=1,inplace=True)
for data in dataset:
    data["Fare"].fillna(data["Fare"].mean(),inplace=True)
test["Cabin"].value_counts()
for data in dataset:
    data["Cabin"] = data["Cabin"].str[:1]
cabin_mapping={"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8} #This is called feature scaling, please explore more on this advanced topic
for data in dataset:
    data["Cabin"] = data["Cabin"].map(cabin_mapping)
for data in dataset:
    data["Cabin"].fillna(data.groupby("Pclass")["Cabin"].transform("median"),inplace=True)
for data in dataset:
    data["Embarked"].fillna(data["Embarked"].mode()[0],inplace=True)
#Label encoding Embarked
for data in dataset:
    data["Embarked"]=le.fit_transform(data["Embarked"])
#for train dataset
sns.heatmap(train.isnull(),cmap = 'magma' )
#For test dataset
sns.heatmap(test.isnull(),cmap = 'magma' )
#separating labels
y_train = train["Survived"]
#separating features
train.drop("Survived", axis=1,inplace=True)
X_train=train
#checking train dataset
X_train.head()
#checking test dataset after removing passengerid as a copy of test data, as we reuire passengerid in final submission on Kaggle
X_test = test.drop("PassengerId",axis=1).copy()
X_test.head()
#Finally check the shapes
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
#import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
#set values for K-folds
folds= KFold(n_splits=10,shuffle=True,random_state=0)
metric="accuracy"
gnb=GaussianNB()
score= cross_val_score(gnb,X_train,y_train,cv=folds,n_jobs=1,scoring=metric)
print(score)
#mean score rounded to 2 decimal points
round(np.mean(score)*100,2)
#using the classifier that gave highest average accuracy on train dataset
clf=GaussianNB()
clf.fit(X_train,y_train)
y_test = clf.predict(X_test)
#wrapping up into a submission dataframe
submission=pd.DataFrame({"PassengerId": test["PassengerId"],"Survived":y_test})

#converting to submission csv
submission.to_csv("submission.csv",index=False)
submission=pd.read_csv("submission.csv")
submission.head()
