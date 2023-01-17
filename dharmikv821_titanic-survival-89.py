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
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
print(train_data.isna().sum())

print(test_data.isna().sum())
print(train_data.nunique())
train_data = train_data.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)

test_data = test_data.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1)
train_data.Embarked = train_data.Embarked.fillna(train_data.Embarked.mode()[0])
print(train_data.isna().sum())

print(test_data.isna().sum())
print(train_data.groupby('Embarked').Age.mean())

print(test_data.groupby('Embarked').Age.mean())
ind = train_data[train_data.Age.isna()].index

for i in ind:

    if train_data.Embarked.loc[i]=='C':

        train_data.Age.loc[i]=30.81

    if train_data.Embarked.loc[i]=='Q':

        train_data.Age.loc[i]=28.09

    if train_data.Embarked.loc[i]=='S':

        train_data.Age.loc[i]=29.44
ind = test_data[test_data.Age.isna()].index

for i in ind:

    if test_data.Embarked.loc[i]=='C':

        test_data.Age.loc[i]=34.74

    if test_data.Embarked.loc[i]=='Q':

        test_data.Age.loc[i]=29.32

    if test_data.Embarked.loc[i]=='S':

        test_data.Age.loc[i]=28.76
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_data.Sex = le.fit_transform(train_data.Sex)

test_data.Sex = le.transform(test_data.Sex)
emb = pd.get_dummies(train_data.Embarked,drop_first=True)

train_data = pd.concat([train_data,emb],axis=1)

emb = pd.get_dummies(test_data.Embarked,drop_first=True)

test_data = pd.concat([test_data,emb],axis=1)

train_data = train_data.drop('Embarked',axis=1)

test_data = test_data.drop('Embarked',axis=1)
train_data.columns
from sklearn.model_selection import train_test_split

X = train_data.iloc[:,1:]

y = train_data.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=101)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

test_data = sc.transform(test_data)
from sklearn.metrics import accuracy_score, confusion_matrix

def accur(data,pred):

    print(accuracy_score(data,pred))

    print(confusion_matrix(data,pred))
from sklearn.svm import SVC

classifier = SVC(random_state=101)

classifier.fit(X_train,y_train)

pred_train = classifier.predict(X_test)

accur(y_test,pred_train)
predictions = classifier.predict(test_data)
pred_data = pd.read_csv('../input/titanic/gender_submission.csv')

accur(pred_data.Survived,predictions)