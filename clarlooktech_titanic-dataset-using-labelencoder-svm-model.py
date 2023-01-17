# Import Libraries



import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder
# Load Dataset From CSV 



titanic_data = pd.read_csv('../input/titanic-dataset/titanic_dataset.csv')

titanic_data.head()
# Get only the data (features) that you will use for analysis and prediction

# We are going to use 'Age', 'Sex','Pclass', 'Fare'



train_data = ['Age', 'Sex','Pclass', 'Fare', 'PassengerId' ]
# load your target data (y)



target_data = ['Survived']
# Put the columns together for analysis and prediction

X = titanic_data[train_data]

Y = titanic_data[target_data]
X.head()
# Identify data type



X['Sex'].dtype
X.Sex
Y.head()
#look for NaN Values



X['Pclass'].isnull().sum()
X['Fare'].isnull().sum()
X['Age'].isnull().sum()
X['Sex'].isnull().sum()
# Since 'Age' has significant number of null values but is an important feature, we are not able to drop this column, 

# But instead, fill Nan Values with 'median' 



X['Age'] = X['Age'].fillna(X['Age'].median())
X['Age'].isnull().sum()
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Fare'].isnull().sum()
# Convert 'Sex' from string into integer using LabelEncoder



le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'].astype(str))

X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size= 0.80)
len(X_train) #how many data are for train
len(X_test) #how many data are for test
from sklearn import svm

model_linearsvc = svm.LinearSVC()
#train the model

model_linearsvc.fit(X_train, y_train)
#check accuracy of model

model_linearsvc.score(X_test, y_test)
from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(X_train, y_train)
model_svc.score(X_test, y_test)
from sklearn import tree
model_tree = tree.DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
model_tree.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
model_lr.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)
# Output array([1]) means the person lived



model_rf.predict(X_test[0:1]) 
model_rf.predict(X_test[0:1]) 
# predict top 10 people of the test dataset



model_rf.predict(X_test[0:223]) 
y_test.head()
submission2 = pd.DataFrame({

        "PassengerId": X_test['PassengerId'],

        "Survived": model_rf.predict(X_test)

    })
submission2
submission2.to_csv('submission2.csv', index=False)