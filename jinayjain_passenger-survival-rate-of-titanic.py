import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#loading train csv file
data=pd.read_csv('../input/titanic/train.csv')
data.head(10)                 
#shape is function to identify the rows and columns
data.shape
#checking the null values
data.isnull().sum()
#we are filling the null values by taking the mean of Age
data['Age']=data['Age'].fillna(data['Age'].mean())
#dropping the cabin column because it is not usable
data.drop('Cabin',axis=1,inplace=True)
#This function helps to drop all the null values present in the column
data.dropna(inplace=True)
#we see that are nan values are filled
data.isnull().sum()
#Data Analysis
sns.countplot(x='Survived',data=data)
sns.countplot(x='Survived',hue='Sex',data=data)
sns.countplot(x='Survived',hue='Pclass',data=data)
data['Age'].plot.hist()
sns.countplot(x='Parch',data=data)
#Data preprocessing
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data['Embarked']=label.fit_transform(data['Embarked'])
data['Embarked'].value_counts()
data['Sex']=label.fit_transform(data['Sex'])
data['Sex'].value_counts()
#we will drop column which is not use
data.drop(['PassengerId','Name','Sex','Embarked','Ticket','Pclass'],axis=1,inplace=True)
#Independent and Dependent variable for prediction
x=data.drop('Survived',axis=1)
y=data['Survived']
#Splitting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
classification_report(y_test,y_pred)
accuracy_score(y_test,y_pred)