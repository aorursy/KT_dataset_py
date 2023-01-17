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
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/titanic-dataset-with-logistic-regression/train (1).csv")
train.head()
train.shape
train.info()
train.describe()
# Checking for null values
train.isnull().sum()
# cabin column has too many null values, and no role in prediction, hence dropping it
train = train.drop('Cabin', axis = 1)
train.head()
# replacing null values of age column by median value of the column
train['Age'] = train['Age'].fillna(train['Age'].median())
train.isna().sum()
train.dropna(inplace = True)
train.isna().sum()
train.info()
# creating dummies for categorical data -> sex, embarked, name doesnt really matter here
sex = pd.get_dummies(train['Sex'], drop_first = True)

embarked = pd.get_dummies(train['Embarked'], drop_first = True)
train = pd.concat([train, sex, embarked], axis = 1)
# dropping useless columns

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
train.head()
sns.countplot(x = 'Survived', data = train)
# count of people not surviving is larger 
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')
# count of people surviving are majorly from the higher class
sns.distplot(train['Age'], bins = 30, kde = False)
# major passengers are of 15-50 age group
X = train.drop('Survived',axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)
test = pd.read_csv("../input/titanic-dataset-with-logistic-regression/test.csv")
test.shape
test.isna().sum()
test = test.drop('Cabin', axis = 1)
test.dropna(inplace = True)
test.info()
test.isna().sum()
test.isnull().sum()
sex = pd.get_dummies(test['Sex'], drop_first=True)

embarked = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sex, embarked], axis = 1)
test.drop(['Sex','Embarked','Name','Ticket', 'PassengerId'], axis = 1, inplace = True)
test.head()
# now test data is prepared, so testing the model on test data
y_pred = model.predict(test)
from sklearn.metrics import confusion_matrix
y_pred_split = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred_split)

sns.heatmap(confusion_matrix(y_test, y_pred_split), annot=True, fmt='3.0f')

plt.title('cm', y=1.05, size=15)