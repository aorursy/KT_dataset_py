# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.preprocessing as preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



import os

bp='/kaggle/input/titanic'

print(os.listdir(bp))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv(bp+'/train.csv')

train_df.head()
test_df=pd.read_csv(bp+'/test.csv')

IDtest = test_df['PassengerId']

test_df.head()

submission_df=pd.read_csv(bp+'/gender_submission.csv')

submission_df.head()
print('elements in sub data : ',submission_df.shape)

print('elements in train data : ',train_df.shape)

print('elements in test data : ',test_df.shape)
total = train_df.isnull().sum().sort_values(ascending= False)

percent= (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing_train_data=pd.concat([total,percent], axis=1, keys=['total','percent'])

missing_train_data
total = test_df.isnull().sum().sort_values(ascending= False)

percent= (test_df.isnull().sum()/test_df.isnull().count()*100).sort_values(ascending = False)

missing_test_data=pd.concat([total,percent], axis=1, keys=['total','percent'])

missing_test_data
ports = pd.get_dummies(train_df.Embarked , prefix='Embarked')

train = train_df.join(ports)

train.drop(['Embarked'], axis=1, inplace=True)

train.Sex = train_df.Sex.map({'male':0, 'female':1})

y = train.Survived.copy()

X = train.drop(['Survived'], axis=1) 

X.drop(['PassengerId'],axis=1,inplace=True)

X.drop(['Cabin'],axis=1, inplace=True)

X.drop(['Ticket'],axis=1, inplace=True)

X.drop(['Name'],axis=1, inplace=True)



X.Age.fillna(X.Age.median(), inplace=True) 
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33, random_state=40)

print('The length of training data : ',X_train.shape, y_train.shape)

print('The length of testing data : ',X_test.shape, y_test.shape)
model= LogisticRegression(max_iter=2)

model.fit(X_train, y_train)



predictions= model.predict(X_test)

predictions
from sklearn.metrics import accuracy_score

print('accuracy score of : ',accuracy_score(predictions, y_test))
subs=pd.Series(model.predict(X_test), name='Survived')

results= pd.concat([IDtest, subs],axis=1)

results.to_csv("Final Submission File.csv",index=False)

results.head(5)