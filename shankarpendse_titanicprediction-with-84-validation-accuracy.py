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
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.isnull().sum()
test.isnull().sum()
train.describe(include=['O'])
dataset = [train,test]

for data in dataset:

    data.drop(['Ticket','Cabin'], axis = 1, inplace = True)
null_cols = ['Age','Embarked','Fare']

for data in dataset:

    data['Age'].fillna(data['Age'].median(), inplace = True)

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    data['Fare'].fillna(data['Fare'].median(), inplace = True)
train.isnull().sum()
test.isnull().sum()
for data in dataset:

    data['Title'] = data['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
train['Title'].value_counts()
test['Title'].value_counts()
train_title_counts = train['Title'].value_counts() < 10

test_title_counts = test['Title'].value_counts() < 10
Misc_count = 10

train['Title'] = train['Title'].apply(lambda x: 'Misc' if train_title_counts.loc[x] == True else x)

test['Title'] = test['Title'].apply(lambda x: 'Misc' if test_title_counts.loc[x] == True else x)
train['Title'].value_counts()
test['Title'].value_counts()
train.columns
test.columns
for data in dataset:

    data['Family_size'] = data['SibSp'] + data['Parch'] + 1

    data['Is_alone'] = 1

    data['Is_alone'].loc[data['Family_size'] > 1] = 0
train['Is_alone'].value_counts()
test['Is_alone'].value_counts()
train['Family_size'].value_counts()
test['Family_size'].value_counts()
for data in dataset:

    data['Fare_bin'] = pd.qcut(data['Fare'], 4)

    data['Age_bin'] = pd.cut(data['Age'].astype(int), 5 )
train.head()
test.head()
train.drop(['PassengerId','Name'], axis = 1, inplace = True)

test.drop(['Name'], axis = 1, inplace = True)
train.head()
test.head()
#for data in dataset:

 #   data.drop(['SibSp','Parch','Age','Fare'], axis = 1, inplace = True)

    

for data in dataset:

    data.drop(['Age','Fare'], axis = 1, inplace = True)
train.head()
test.head()
cat_attr = ['Sex','Title','Embarked','Fare_bin','Age_bin']
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for data in dataset:

    for attr in cat_attr:

        data[attr] = encoder.fit_transform(data[attr])

    
train.head()
test.head()
attr = ['Pclass','Sex','Embarked','Family_size','Is_alone','Fare_bin','Age_bin','SibSp','Parch','Title']

for col in attr:

    print(train[['Survived',col]].groupby(col).mean())

    print()
train.corr()
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(train.drop(['Survived','Family_size'],axis = 1),train['Survived'],test_size = 0.2, shuffle = True,stratify = train['Survived'],random_state = 42)
X_train.shape
X_train.columns
X_val.shape
X_val.columns
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
DTC = DecisionTreeClassifier()

RFC = RandomForestClassifier()

NBC = GaussianNB()

SVMC = SVC()

LRC = LogisticRegression()

XGBC = XGBClassifier()
MLAs = [DTC,RFC,NBC,SVMC,LRC,XGBC]
train.columns
test.columns
X_train.columns
X_val.columns
test.columns
for algo in MLAs:

    algo.fit(X_train,y_train)

    print("Algorithm: ",algo)

    print("Training accuracy: ", algo.score(X_train,y_train))

    print("validation accuracy: ", algo.score(X_val,y_val))

    print("**************************************\n")
RFC = RandomForestClassifier(n_estimators = 300, criterion = 'gini', max_depth = 4, random_state = 42)
RFC.fit(X_train,y_train)

print("Training accuracy: ", RFC.score(X_train,y_train))

print("validation accuracy: ", RFC.score(X_val,y_val))
test.head()
X_train.head()
test.drop(['Family_size'], axis = 1, inplace = True)
X_train.columns
test.columns
predictions = RFC.predict(test.drop(['PassengerId','Survived'],axis = 1))
len(predictions)
test.shape
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = predictions
test.head()
test['Survived'] = predictions
test.head()
del submit
submit = test[['PassengerId','Survived']]
submit.shape
submit['Survived'].value_counts(normalize=True)
submit.sample(10)
submit.to_csv("../working/submit.csv", index=False)