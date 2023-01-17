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
train=pd.read_csv('/kaggle/input/titanic/train.csv')  #reading the train data
train.head(10)                                        #printing out the first 10 rows
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"] # collecting all features 
#separating the features-X from the target variable-y
X=train[features]
y=train["Survived"]
X.isnull().sum()  #figuring out the number of null values in the features
#Cleaning the data by handling missing values from Age and Embarked Columns
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Embarked']=X['Embarked'].fillna(X['Embarked'].mode())
X
X.info()    # Getting the information so as to analyse which columns in the dataset have object type values which have to be fixed by LabelEncoding in the preprocessing step
#Label Encoding or Preprocessing step
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
X['Sex']=Le.fit_transform(X['Sex'])
X['Embarked']=Le.fit_transform(X['Embarked'].astype(str))
X
#Splitting the train data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

# Fitting a Logistic regression model on the training data
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
log.score(X_test,y_test)   #The training data model gives an accuracy of 81%
test=pd.read_csv('/kaggle/input/titanic/test.csv')  #reading the test data
test.head(10)
test_data_x=test[features]    #Repeating the same procedure on test data by making a feature set
test_data_x
# The Cleaning Phase for test data
test_data_x['Age'] = test_data_x['Age'].fillna(test_data_x['Age'].median())
test_data_x['Embarked']=test_data_x['Embarked'].fillna(test_data_x['Embarked'].mode())
test_data_x
#The Preprocessing Phase
test_data_x['Sex']=Le.fit_transform(test_data_x['Sex'])
test_data_x['Embarked']=Le.fit_transform(test_data_x['Embarked'].astype(str))
test_data_x
test_data_x.info()
test_data_x.isnull().sum()
#Cleaning the Fare column as it has 1 null value
test_data_x['Fare']=test_data_x['Fare'].fillna(test_data_x['Fare'].median())
test_data_x
#Making Predictions on the test data 
predictions=log.predict(test_data_x)
predictions
output=pd.DataFrame({"PassengerId":test['PassengerId'],"Survived":predictions})
output.to_csv("submission.csv",index=False)

output.head(10)   #Hence the output