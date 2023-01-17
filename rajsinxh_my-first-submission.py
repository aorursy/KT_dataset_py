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
#importing the basic libraries

import pandas as pd

import numpy as np
#importing the training dataset

data = pd.read_csv("../input/titanic/train.csv")
#view the dataset

data.head()
# removing the useless contents from the dataset

data = data.drop("Name", axis=1)

data = data.drop("Ticket", axis=1)

data = data.drop("Fare", axis=1)

data = data.drop("Cabin", axis=1)
data.head()
data.describe()
#we need to make Parch and SibSp into one column

data["parents"] = data["SibSp"] + data["Parch"]

data.head()
#drop both the columns

data.drop("SibSp",axis = 1,inplace =True)
data.drop("Parch",axis=1,inplace=True)
data.head()
data.isna().sum()
#converting the object types into string type

data["Sex"].dtype
data["Sex"] = data["Sex"].astype('str')
data["Embarked"] = data["Embarked"].astype('str')
data["Embarked"].dtype
#encoding categorical values into numerical ones

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



data["Sex"] = le.fit_transform(data["Sex"])

data["Embarked"] = le.fit_transform(data["Embarked"])
#create test series

Sex = pd.Series(["male", "female", "male"])

transformed = le.fit_transform(Sex)
data.head()
data.isna().sum()
data = data.fillna(data["Sex"].mean())
data.isna().sum()
# rearranging the columns

data = data[["PassengerId", "Pclass" , "Sex", "Age", "Embarked", "parents", "Survived"]]

data.head()
X = data.iloc[:,:-1].values

y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
#importing all the classifier 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import svm,tree

import xgboost
#initializing all the classifiers and creating a constructor of the models

classifiers=[]



model1 = RandomForestClassifier()

classifiers.append(model1)



model2 = xgboost.XGBClassifier()

classifiers.append(model2)



model3 = svm.SVC()

classifiers.append(model3)



model4 = tree.DecisionTreeClassifier()

classifiers.append(model4)



model5 = GradientBoostingClassifier()

classifiers.append(model5)
#fitting our algorithm in classifier array

from sklearn.metrics import accuracy_score, confusion_matrix

for clf in classifiers:

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)

    print("accuracy of the %s is %s"%(clf,acc))

    cm = confusion_matrix(y_test,y_pred)

    print("Confusion matrix of the %s is %s"%(clf,cm))
#importing our test data

data1 = pd.read_csv("../input/titanic/test.csv")
data1.head()
data1 = data1.drop("Name", axis=1)

data1 = data1.drop("Ticket", axis=1)

data1 = data1.drop("Fare", axis=1)

data1 = data1.drop("Cabin", axis=1)
data1.head()
#we need to make Parch and SibSp into one column

data1["parents"] = data1["SibSp"] + data1["Parch"]

data1.head()
#drop both the columns

data1.drop("SibSp",axis = 1,inplace =True)

data1.drop("Parch",axis=1,inplace=True)
data1.head()
#converting bject type into str

data1["Sex"] = data1["Sex"].astype('str')

data1["Embarked"] = data1["Embarked"].astype('str')
#encoding categorical values into numerical ones

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



data1["Sex"] = le.fit_transform(data1["Sex"])

data1["Embarked"] = le.fit_transform(data1["Embarked"])
#create test series

Sex = pd.Series(["male", "female", "male"])

transformed = le.fit_transform(Sex)
#removing the null values

data1.isna().sum()
data1 = data1.fillna(data["Sex"].mean())
data1.isna().sum()
data1
y_preds = model5.predict(data1)
y_preds
data1["survival"] = y_preds
output = pd.DataFrame({'PassengerId': data1.PassengerId, 'Survived': y_preds})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")