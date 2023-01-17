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
import pandas as pd 

train = pd.read_csv("../input/titanic/train.csv")



test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
train.info()
test.info()
train = train.drop(["Cabin","Name", "Ticket"], axis=1)
test = test.drop(["Cabin","Name", "Ticket"], axis=1)
train["Age"].mean()
train["Fare"].mean()
train["Fare"] = train["Fare"].replace(np.nan, 32)

test["Fare"] = test["Fare"].replace(np.nan, 32)

train["Age"] = train["Age"].replace(np.nan, 30)

test["Age"] = test["Age"].replace(np.nan, 30)

train["Embarked"] = train["Embarked"].replace(np.nan, "C")

train.info()
test.info()
train["Sex"].replace(["female","male"] , [0,1], inplace = True)

test["Sex"].replace(["female","male"] , [0,1], inplace = True)

train["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)

test["Embarked"].replace(["Q","S","C"],[0,1,2],inplace=True)
bins = [0,8,15,20,40,60,100]

names=(['Baby', 'Child', 'Teenager', 'Youngster', 'Adult', 'Senior Citizen'])



train["Age"] = pd.cut(train["Age"], bins, labels = names)

test["Age"] = pd.cut(test["Age"], bins, labels = names)
train["Fare"] = pd.cut(train.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])

test["Fare"] = pd.cut(test.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])
train.pivot_table(index = "Sex", values = "Survived")
train.pivot_table(index = "Pclass", values = "Survived")
train.pivot_table(index = "Age", values = "Survived")
sns.barplot(x="Sex", y="Survived", data=train)

plt.show()
sns.barplot(x="Pclass", y="Survived", data=train)

plt.show()
sns.barplot(x="Age", y="Survived", data=train)

plt.show()
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Youngster': 4, 'Adult': 5, 'Senior Citizen': 6}

train['Age'] = train['Age'].map(age_mapping)

test['Age'] = test['Age'].map(age_mapping)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age', 'Fare']
from sklearn.model_selection import train_test_split
test_df = test

X = train[columns]

y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)
from sklearn.metrics import accuracy_score

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)
from sklearn.model_selection import cross_val_score



lr = LogisticRegression()

scores = cross_val_score(lr, X, y, cv=10)

accuracy = np.mean(scores)

print(scores)

print(accuracy)
#Final Model

columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age', 'Fare']

lr = LogisticRegression()

lr.fit(X,y)

test_df_predictions = lr.predict(test_df[columns])
#Submission dataframe

test_df_ids = test_df["PassengerId"]

submission_df = {"PassengerId": test_df_ids,

                 "Survived": test_df_predictions}

submission = pd.DataFrame(submission_df)

submission.head()
submission.to_csv("submission.csv",index=False)





print(lr.score(X_test, y_test))