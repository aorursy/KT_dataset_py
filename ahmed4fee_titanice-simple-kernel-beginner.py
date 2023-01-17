import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train.info()
train.isnull().sum()


plt.figure(figsize=(12,10))

sns.heatmap(train.corr(), annot=True ,cmap='coolwarm')

sns.countplot(x="Survived" , hue="Sex" , data=train)
sns.catplot(x="Pclass",y= "Survived"  ,kind="box", data=train)
sns.boxplot(x="Survived" ,y= "Age", data=train)
sns.barplot(x="Survived" ,y= "SibSp", data=train)
train['Age'].hist(bins=30,alpha=0.7)
train.isnull().sum()


train["Age"].fillna(train["Age"].mean() , inplace=True)

train["Embarked"]=train["Embarked"].fillna("S")
train=train.drop(columns=["Cabin" , "Name","PassengerId","Ticket"] )
train.isnull().sum()
train.info()
train.columns
train.head()
labelencoder=LabelEncoder()

train["Sex"]=labelencoder.fit_transform(train["Sex"])

train["Embarked"]=labelencoder.fit_transform(train["Embarked"])



train.head()
y=train["Survived"]



#print(y)

train.columns

train=train.drop(columns=["Survived"])

train.columns


scaler=StandardScaler()

X=train

X=scaler.fit_transform(X)

print(X)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=41)
lr=LogisticRegression()

lr.fit(X_train,y_train)
y_pred =lr.predict(X_valid)
print(lr.intercept_)
print(lr.coef_)
mae= mean_squared_error(y_valid,y_pred)

mae

test.head()
test=test.drop(columns=["Cabin" , "Name","Ticket"] )
#test_df=pd.DataFrame(test_df)  
test.isnull().sum()


test["Age"].fillna(test["Age"].mean() , inplace=True)

test["Fare"].fillna(test["Fare"].mean() , inplace=True)

test.isnull().sum()
test["Sex"]=labelencoder.fit_transform(test["Sex"])

test["Embarked"]=labelencoder.fit_transform(test["Embarked"])
test.head()
test_df=test.drop(columns="PassengerId")
test_df=scaler.fit_transform(test_df)
prediction=lr.predict(test_df)
print(prediction)
output=pd.DataFrame({"PassengerId":test.PassengerId ,"Survived":prediction})
output.head()
output.to_csv("Submission_csv",index=False)