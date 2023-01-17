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
import pandas as pd
import numpy as np
#For creating the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
train.info()
test.info()
train.isnull().sum()
test=test.drop('Cabin',axis=1)
train=train.drop('Cabin',axis=1)

test['Age'] = test['Age'].fillna(test['Age'].median())
train.dropna(inplace=True)

embark_dummied = pd.get_dummies(test['Embarked'],drop_first=True)
sex_dummied = pd.get_dummies(test['Sex'],drop_first=True)

test = pd.concat([test,sex_dummied,embark_dummied],axis=1)
passenger_id=test["PassengerId"]
test.drop(['Name','Sex','PassengerId','Ticket','Fare','Embarked'],axis=1,inplace=True)
train.drop(['Name','Sex','PassengerId','Ticket','Fare','Embarked'],axis=1,inplace=True)
test.head()

y = train['Survived']
train.head()

X = train.drop('Survived',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
print("testing accuracy",classifier.score(X_train,y_train))
print("training accuracy",classifier.score(X_test,y_test))


y_pred=classifier.predict(X_test)
Confusion_matrix=confusion_matrix(y_test,y_pred)
print(Confusion_matrix)