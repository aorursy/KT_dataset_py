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
data=pd.read_csv("../input/titanic/train.csv")
data.info()
data1=data.drop(['Cabin','Name','Ticket'], axis = 1) 
data1.info()
data1.head()
data1[data1['Age'].isnull()]
data1.Age.fillna(data1['Age'].mean(), inplace=True)
data1.Embarked.fillna(data1['Embarked'].mode()[0], inplace=True)
data1.info()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data1.Sex = le.fit_transform(data1.Sex)

data1.head()
le = LabelEncoder()

data1.Embarked = le.fit_transform(data1.Embarked)

data1.head()
from sklearn.linear_model import LogisticRegression
X = data1.drop('Survived', axis = 1) 





y = data1['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)
model = LogisticRegression()



model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import f1_score

print("F1 Score: ",f1_score(y_test, pred))
test_new= pd.read_csv("../input/titanic/test.csv")
test_new.info()
predictiontab=test_new.drop(['Cabin','Name','Ticket'], axis = 1) 
predictiontab.info()
predictiontab.Age.fillna(predictiontab['Age'].mean(), inplace=True)
predictiontab.Fare.fillna(predictiontab['Fare'].mean(), inplace=True)
le = LabelEncoder()

predictiontab.Embarked = le.fit_transform(predictiontab.Embarked)

predictiontab.head()
le = LabelEncoder()

predictiontab.Sex = le.fit_transform(predictiontab.Sex)

predictiontab.head()
Answer = model.predict(predictiontab)
Answer