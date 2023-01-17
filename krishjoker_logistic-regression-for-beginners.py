import pandas as pd
import numpy as np
train=pd.read_csv("../input/train.csv")
train.head(2)
##information from train dataset
train.info()
#Dropping Cabin Column for more than 70 % of missing points
train.drop('Cabin',axis=1,inplace=True)
##Filling Missing Values Based on Mean
train.fillna(round(train.mean(),0), inplace=True)
##Dropping only Two rows in Embarked which is having Two missing Values
train.dropna(axis=0,inplace=True)
from sklearn.preprocessing import LabelEncoder
#Selecting only Numeric Columns
train_numeric=train.select_dtypes(exclude=['object'])
#Selecting Only Object Columns
train_categorical=train.select_dtypes(include=["object"])

##Making categorical columns into numerical ones

lbmake=LabelEncoder()
b=["Name","Sex", "Ticket","Embarked"]
for i in range(0,len(b)):
    train_categorical[b[i]]= lbmake.fit_transform(train_categorical[b[i]])
print(train_categorical.head(3))
#Droping Name and Ticket number for model not needed
train_categorical.drop(["Name","Ticket"],axis=1,inplace=True)
#Merging of Datasets
train_last=pd.concat([train_categorical,train_numeric],axis=1)
train_last.drop("PassengerId",axis=1,inplace=True)
train_last.head(3)
from sklearn.linear_model import LogisticRegression
y_train=train_last.iloc[:,2:3]
y_train.head(3)
x_train=train_last.iloc[:,[0,1,2,4,5,6,7]]
x_train.head(3)
##Model Fitting
logisticRegr=LogisticRegression()
log_model=logisticRegr.fit(x_train, y_train)
##Testing Process
test=pd.read_csv("../input/test.csv")
test.drop('Cabin',axis=1,inplace=True)
test.fillna(round(test.mean(),0), inplace=True)
test.dropna(axis=0,inplace=True)
test_numeric=test.select_dtypes(exclude=['object'])
test_categorical=test.select_dtypes(include=["object"])
lbmake=LabelEncoder()
b=["Name","Sex", "Ticket","Embarked"]
for i in range(0,len(b)):
    test_categorical[b[i]]= lbmake.fit_transform(test_categorical[b[i]])
print(test_categorical.head(3))

test_last=pd.concat([test_categorical,test_numeric],axis=1)
test_last.drop(["PassengerId","Ticket","Name"],axis=1,inplace=True)

##Training of Model using Train set
from sklearn.linear_model import LogisticRegression
log_make=LogisticRegression()
log_model=log_make.fit(x_train,y_train)

#Testing of Model using test set
test["Survived"]=log_model.predict(test_last)


##For  Submit to Check Prediction
##gen_sub1=pd.read_csv("C:/Users/Selvamani/Desktop/gen_sub1.csv")
##gen_sub1["Survived"]=test["Survived"]
##gen_sub1.to_csv("C:/Users/Selvamani/Desktop/gen_sub2.csv")