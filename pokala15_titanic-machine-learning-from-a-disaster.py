#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing the dataset
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
y_test=pd.read_csv("../input/gender_submission.csv")
#top five elements of traning dataset
train.head()
#droping the columns which are not required for the prediction
train=train.drop(['PassengerId','Name','Ticket'],axis=1)
train.head()
#one hot encoding for coloumn Sex
d_t=pd.get_dummies(train['Sex'])
d_t
d_t.drop(['male'],axis=1)
train=train.join(d_t)
train=train.drop(['Sex'],axis=1)
train.head()
#filling the missing values in the specified coloumns
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Cabin'].fillna(train['Cabin'].mode(),inplace=True)
train=train.drop(['male'],axis=1)
train.head()
#one hot encoding for Embarked coloumn
e_t=pd.get_dummies(train['Embarked'])
e_t=e_t.drop(['S'],axis=1)
e_t
train=train.drop(['Embarked'],axis=1)
train=train.join(e_t)
train=train.drop(['Cabin'],axis=1)
train.head()
#converting data frame to numpy array
x=train.iloc[:,1:9].values
y=train.iloc[:,0:1].values
#creating logistic regression model
from sklearn.linear_model import LogisticRegression
le=LogisticRegression()
le.fit(x,y)
test.head()
#droping the coloumns which are not required
test=test.drop(['Name','PassengerId','Ticket'],axis=1)
test.head()
test=test.drop(['Cabin'],axis=1)
test.head()
#one hot encoding for the specified coloumns
st=pd.get_dummies(test['Sex'])
st=st.drop(['male'],axis=1)
test=test.drop(['Sex'],axis=1)
test=test.join(st)
test.head()
et=pd.get_dummies(test['Embarked'])
et=et.drop(['S'],axis=1)
test=test.drop(['Embarked'],axis=1)
test=test.join(et)
test.head()
test.isnull().sum()
#filling the missing values
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.isnull().sum()
test.head()
train.head()
#conveting test data frame to numpy array
x_test=test.iloc[:,:].values
#converting datatype float to int
test['Age']=test['Age'].astype(int)
test['Fare']=test['Fare'].astype(int)
#predicting output for the given test set
y_pred=le.predict(x_test)
y_t=y_test.iloc[:,1:2].values
#checking for accuracy
from sklearn.metrics import confusion_matrix
ob=confusion_matrix(y_t,y_pred)
ob
#creating model with SVC
from sklearn.svm import SVC 
s=SVC(kernel='linear',random_state=0)
s.fit(x,y)
#predicting the output using the SVC model
y_p=s.predict(x_test)
#accuracy for the model obtained from SVC
from sklearn.metrics import confusion_matrix
ob=confusion_matrix(y_t,y_p)
ob