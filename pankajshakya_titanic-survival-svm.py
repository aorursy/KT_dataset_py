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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

%notebook inline
#load data

train=pd.read_csv('../input/titanic/train.csv')

test =pd.read_csv("../input/titanic/test.csv")
train.describe()
train.info()
train.head()
test.head()
## Drop some of the colum which is not useful

train = train.drop(columns = ["Name","Cabin","Ticket"])

test = test.drop(columns = ["Name","Cabin","Ticket"])
train.head()
test.head()
print(train.shape)

print(test.shape)
##Check Missing value

train.isnull().sum()

sns.countplot(train['Sex'], hue = "Survived",data=train)
null_columns=train.columns[train.isnull().any()]



train[null_columns].isnull().sum()
# train[train["Age"].isnull()][null_columns]
## fill the missing value using median

train['Age'] = train['Age'].replace(np.NaN, train['Age'].median())
print(train[train["Age"].isnull()][null_columns])
# fill the missing value

mode_value=train['Embarked'].mode()[0]

train['Embarked']=train['Embarked'].fillna(mode_value)
# Find the Missing values

null_columns=test.columns[test.isnull().any()]



test[null_columns].isnull().sum()
test['Fare'] = test['Fare'].replace(np.NaN, train['Fare'].mean())
test['Age'] = test['Age'].replace(np.NaN, train['Age'].median())

mode_value=test['Embarked'].mode()[0]

test['Embarked']=test['Embarked'].fillna(mode_value)
print(train.isnull().sum())

print("__^_^__"*10)

print(test.isnull().sum())
train['Sex'] = train["Sex"].replace({'male': 1, 'female': 0})
# train['Sex']
test['Sex'] = test["Sex"].replace({'male': 1, 'female': 0})
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 



train['Embarked']= le.fit_transform(train['Embarked']) 

test['Embarked'] = le.fit_transform(test['Embarked']) 
# test["Embarked"]
# train['Embarked'].unique()
# test['Embarked'].unique()
train.head()
test.head()
X = train.drop(columns = ["Survived"])

y = train['Survived']
X.head()
y
from sklearn.svm import SVC
svm =  SVC(kernel="rbf", C=0.025,random_state=101)

svm.fit(X, y)

y_pred=svm.predict(test)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")