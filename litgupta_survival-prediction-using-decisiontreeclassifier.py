 #This is my first effort in Kaggle, any suggestion is welcome :)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#loading data

train_file=pd.read_csv('../input/train.csv')

test_file=pd.read_csv('../input/test.csv')
#checking the columns of the data

print (train_file.columns)

print (test_file.columns)
#replacing the textual values with integer value

train_file['Sex']=train_file['Sex'].map({'female':1,'male':0})

test_file['Sex']=test_file['Sex'].map({'female':1,'male':0})
#Filling the null values with average values

train_file['Age']=train_file['Age'].fillna(train_file['Age'].median())

test_file['Age']=test_file['Age'].fillna(test_file['Age'].median())
#Generating the testing features and targets for our machine learning model

train_features=train_file[["Pclass","Age","Sex"]].values

target=train_file["Survived"].values
from sklearn.tree import DecisionTreeClassifier #Using DecesionTreeClassifier for classification task

from sklearn.cross_validation import train_test_split #for splitting the training data to test accuracy

from sklearn.metrics import accuracy_score #to test accuracy
model=DecisionTreeClassifier()

x_train,x_test,y_train,y_test=train_test_split(train_features,target)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)
accuracy_score(y_test,y_predict)
#generating the testing-features from test_file

test_features=test_file[["Pclass","Age","Sex"]].values
test_answer=model.predict(test_features)
PassengerId=np.array(test_file["PassengerId"]).astype(int)

solution=pd.DataFrame(test_answer,PassengerId,columns=["Survived"])
#checking for exact 418 entries

solution.shape
#Generating the solution_csv file for submission

solution.to_csv("Solution_one.csv",index_label=["PassengerId"])