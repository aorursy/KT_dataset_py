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

import math as mt

import json as js

import seaborn as sns
submis_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

submis_data
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data
train_data['Sex'].replace(['male','female'],['1','0'],inplace=True)
train_data.info()
train_data.drop('Cabin',axis=1,inplace=True)
train_data.info()
embrk = train_data['Embarked'].value_counts().idxmax()

train_data['Embarked'].fillna(embrk,inplace=True)
age = train_data['Age'].median(skipna=True)

train_data['Age'].fillna(age,inplace=True)
train_data.info()
sns.countplot(x='Embarked',order=['Q','C','S'],data=train_data)
train_data['Embarked'].replace(['Q','C','S'],['1','2','3'],inplace=True)
train_data.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)

train_data
sns.set(rc={'figure.figsize':(20,10)})

sns.countplot(x='Age',data=train_data)
result = train_data['Survived']

train_data.drop('Survived',axis=1,inplace=True)
from sklearn.model_selection import train_test_split



x=train_data.values

y=result.values



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=False, random_state=1

                            , verbose=0,

                       warm_start=False)

rf.fit(x_train,y_train)
y_test_pred = rf.predict(x_test)



y_train_pred = rf.predict(x_train)

print(accuracy_score(y_train,y_train_pred))

print(accuracy_score(y_test,y_test_pred))

last_rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=4, max_features='auto',

                       max_leaf_nodes=5, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=15,

                       min_weight_fraction_leaf=0.0, n_estimators=350,

                       n_jobs=None, oob_score=True, random_state=1

                            , verbose=0,

                       warm_start=False)

last_rf.fit(train_data,result)

last_rf.oob_score_

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

test['Embarked'].replace(['Q','C','S'],['1','2','3'],inplace=True)

test['Sex'].replace(['male','female'],['1','0'],inplace=True)

test
test.info()
age = test['Age'].median(skipna=True)

test['Age'].fillna(age,inplace=True)

test.info()
age = test['Fare'].median(skipna=True)

test['Fare'].fillna(age,inplace=True)

test.info()
ids = test['PassengerId']

test.drop(['PassengerId'],axis=1,inplace=True)



y_test = last_rf.predict(test)

y_test
submission = pd.DataFrame(y_test,index=ids,columns=['Survived'])

submission.to_csv('gender_submission.csv')
cp /kaggle/working/submission.csv /kaggle/working/gender_submission.csv
rm -rf /kaggle/working/submission.csv