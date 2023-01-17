# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#loading the train dataset
data=pd.read_csv("../input/titanic/train.csv")
#loading test dataset
data_test=pd.read_csv('../input/titanic/test.csv')
#Let's quickly have a look on data
data.head()
#analysing some features of data
data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)

data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)

data[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Let's have a summary of dataset
data.describe()
data.describe(include=['O'])
#some visualisation
data['Age'].plot.hist()

sns.countplot(x='Survived',hue='Sex',data=data)
#data wrangling
data.drop(['Name','Ticket','PassengerId','Cabin'],axis= 1,inplace=True)
data.head()

data.isnull().sum()
#removing missing values 
data['Embarked'].dropna(inplace=True)
data['Age']=data['Age'].fillna(30)
#adding dummy variables
gender=pd.get_dummies(data['Sex'],drop_first=True)
pclass=pd.get_dummies(data['Pclass'],drop_first=True)
embarked=pd.get_dummies(data['Embarked'],drop_first=True)
#adding some new columns and removing some unwanted columns
data=pd.concat([data,gender,pclass,embarked],axis=1)
data.drop(['Pclass','Sex','Embarked'],axis=1,inplace=True)
data.head()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#splitting data into training and testing dataset
x1=data.drop(['Survived','Parch','Fare'],axis=1)
y1=data['Survived']
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.20,random_state=42)
#fitting logisting model to the dataset
Lreg=LogisticRegression(solver='lbfgs')
Lreg.fit(x1_train,y1_train.ravel())
y1_predict=Lreg.predict(x1_test)
y1_predict
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
x11=sm.add_constant(x1)
logit_model=sm.Logit(y1.astype(float),x11.astype(float))
result1=logit_model.fit()
print(result1.summary())
#calculating accuracy
from sklearn.metrics import accuracy_score
score1= accuracy_score(y1_test,y1_predict)
score1
#forming confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y1_test,y1_predict)
#now lets have a look on test data
data_test.head()
#analysing data
data_test.describe()
data_test.describe(include=['O'])
#data wrangling
data_test['Age']=data_test['Age'].fillna(30)

data_test.isnull().sum()
data_test.drop(['Name','Ticket','Fare','Cabin'],inplace=True,axis=1)
#creating dummy variables
gender_test=pd.get_dummies(data_test['Sex'],drop_first=True)
pclass_test=pd.get_dummies(data_test['Pclass'],drop_first=True)
embarked_test=pd.get_dummies(data_test['Embarked'],drop_first=True)
data_test=pd.concat([data_test,gender_test,pclass_test,embarked_test],axis=1)
data_test.drop(['Pclass','Sex','Embarked'],axis=1,inplace=True)
data_test.head()
x_test1=data_test.drop(['PassengerId','Parch'],axis=1)
#predicting the values
y_test_predict=Lreg.predict(x_test1)
y_test_predict
#forming data frame
final={'survived':y_test_predict}
sub=pd.DataFrame(final)
id=data_test['PassengerId']
sub.insert(0,'PassengerId',id,True)
#saving file in csv format
sub.to_csv('file1.csv')