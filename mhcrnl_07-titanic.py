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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test  = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission= pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(train.describe())

print(train.head())
test.describe()                                                                  
gender_submission.describe()
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)

y = train['Survived']

X = pd.get_dummies(train['Sex'])

model.fit(X, y)

test['Survived'] = model.predict(pd.get_dummies(test['Sex']))
from sklearn.metrics import accuracy_score

accuracy_score(gender_submission['Survived'] ,test['Survived'])
test[['PassengerId', 'Survived']].to_csv('07_sub.csv', index=False)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# create a pipeline object

pipe = make_pipeline(StandardScaler(),

                     LogisticRegression(random_state=0))



# load data

X=pd.get_dummies(train['Sex'])

y=train['Survived']



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# fit the whole pipeline

pipe.fit(X_train, y_train)



Pipeline(steps=[('standardscaler',StandardScaler()),

                ('logisticregression', LogisticRegression(random_state=0))])



test['Survived']=pipe.predict(pd.get_dummies(test['Sex']))



# accuracy_score(pipe.predict(X_test),y_test)

                            

test[['PassengerId','Survived']].to_csv("07_LR_sub.csv", index=False)

type(test)
# Fiting and predicting : estimator basics

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(random_state=0)



X=pd.get_dummies(train['Sex'])

y=train['Survived']



clf.fit(X,y)

test['Survived']=clf.predict(pd.get_dummies(test['Sex']))



test[['PassengerId','Survived']].to_csv("07_RFC_sub.csv", index=False)

                             
# Liniar Regression

from sklearn import linear_model



lr = linear_model.LinearRegression()



X=pd.get_dummies(train['Sex'])

y=train['Survived']



lr.fit(X,y)

test['Survived']=lr.predict(pd.get_dummies(test['Sex']))



test[['PassengerId','Survived']].to_csv("07_LR1_sub.csv", index=False)