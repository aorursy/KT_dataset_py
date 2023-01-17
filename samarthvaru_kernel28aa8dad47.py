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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train
train.isnull().sum()
train.corr()
import seaborn as snp
snp.countplot(train['Pclass'])
train['Age'].value_counts()
snp.distplot(a=train['Age'])
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

mypipeline=Pipeline([

    

    ('imputer',SimpleImputer(strategy="median"))

])
train.describe(include="all")



train.dtypes
train=train.drop(['Name','Ticket'],axis=1)
train['Sex'].unique
train['Embarked'].unique
train['Sex']=train['Sex'].map({'male':1,'female':2})

train['Embarked']=train['Embarked'].map({'C':0,'S':1,'Q':2})
train
train=train.drop('Cabin',axis=1)
headers=train.columns.values

headers=list(headers)
train=mypipeline.fit_transform(train)
train
cleaned_train=pd.DataFrame(train)
cleaned_train.columns=headers

cleaned_train
cleaned_train.isnull().sum()
snp.distplot(cleaned_train['Age'],kde=True)
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

test.isnull().sum()
test=test.drop(['Name','Ticket','Cabin'],axis=1)
test['Sex']=test['Sex'].map({'male':1,'female':2})

test['Embarked']=test['Embarked'].map({'C':0,'S':1,'Q':2})
test=mypipeline.fit_transform(test)
cleaned_test=pd.DataFrame(test)

headers.remove('Survived')

cleaned_test.columns=headers

cleaned_test

from sklearn.model_selection import train_test_split



# Here is out local validation scheme!

X_train, X_test, y_train, y_test = train_test_split(cleaned_train.drop(['Survived'], axis = 1), 

                                                    cleaned_train['Survived'], test_size = 0.2, 

                                                    random_state = 42)
model=RandomForestClassifier(random_state=42,n_estimators=500)
model.fit(X_train,y_train)
model.score(X_train,y_train)

pred=model.predict(X_test)

pred
from sklearn.metrics import accuracy_score

print(accuracy_score(pred,y_test))

cleaned_test['Survived']=model.predict(cleaned_test)
cleaned_test['PassengerId']=cleaned_test['PassengerId'].astype(int)
from sklearn.metrics import classification_report, confusion_matrix



print(confusion_matrix(y_test, pred))
cleaned_test[['PassengerId','Survived']]
cleaned_test[['PassengerId','Survived']].to_csv('kaggle_submission.csv',index=False)