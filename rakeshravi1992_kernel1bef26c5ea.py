# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

submission_df = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission_df
train_df.dtypes
train_df["Survived"].hist()
train_df["Pclass"].value_counts()

## CATEGORICAL
train_df["Name"].sample(100).head(20)
pd.pivot_table(train_df,index=["Pclass","Survived"],values=["PassengerId"],aggfunc=[len])
train_df["Embarked"].value_counts()
pd.pivot_table(train_df,index=["Embarked","Survived"],values=["PassengerId"],aggfunc=[len])
train_df.head()
from sklearn.preprocessing import LabelEncoder
l_Pclass = LabelEncoder()

train_df['Pclass'] = l_Pclass.fit_transform(train_df['Pclass'])
l_name = LabelEncoder()

train_df['Name'] = l_name.fit_transform(train_df['Name'])
l_Sex = LabelEncoder()

train_df['Sex'] = l_Sex.fit_transform(train_df['Sex'])
l_Ticket = LabelEncoder()

train_df['Ticket'] = l_Ticket.fit_transform(train_df['Ticket'])
train_df['Cabin'].fillna(train_df['Cabin'].mode()[0], inplace = True)

combined_df = pd.concat([test_df,train_df], axis = 0)

l_C = LabelEncoder()

l_C.fit(combined_df["Cabin"].astype("str"))

test_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)

test_df['Cabin'].fillna(train_df['Cabin'].mode()[0], inplace = True)

train_df['Cabin'] = l_C.fit_transform(train_df['Cabin'])

test_df['Cabin'] = l_C.fit_transform(test_df['Cabin'])
l_Embarked = LabelEncoder()

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)

train_df['Embarked'] = l_Embarked.fit_transform(train_df['Embarked'])
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_df[[ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']], train_df["Survived"], test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=50, random_state=0, max_depth = 6,max_features="log2")
model.fit(X_train, y_train)
y_preds = model.predict(X_val)

accuracy_score(y_val, y_preds)
test_df['Cabin']
test_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)
combined_df = pd.concat([test_df,train_df], axis = 0)

l_C = LabelEncoder()

l_C.fit(combined_df["Cabin"])

test_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)

test_df['Cabin'].fillna(train_df['Cabin'].mode()[0], inplace = True)

train_df['Cabin'] = l_C.fit_transform(train_df['Cabin'])

test_df['Cabin'] = l_C.fit_transform(test_df['Cabin'])
test_df['Pclass'] = l_Pclass.fit_transform(test_df['Pclass'])

test_df['Name'] = l_name.fit_transform(test_df['Name'])

test_df['Sex'] = l_Sex.fit_transform(test_df['Sex'])

# test_df['Cabin'] = l_C.fit_transform(test_df['Cabin'])

test_df['Embarked'] = l_Embarked.fit_transform(test_df['Embarked'])

test_df['Ticket'] = l_Ticket.fit_transform(test_df['Ticket'])
test_df.dtypes
test_df["Survived"] = model.predict(test_df[[ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']])
submission = test_df[["PassengerId","Survived"]]

submission.shape, submission_df.shape
submission.to_csv("submission.csv", index = None)