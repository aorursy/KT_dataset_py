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

from sklearn import linear_model

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import seaborn as sb

%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
test.info()
train.drop(['Name', 'Ticket', 'Cabin'],axis= 1, inplace= True)

test.drop(['Name', 'Ticket', 'Cabin'],axis= 1, inplace= True)
avg_age = train.Age.mean()

train['Age'] = train.Age.fillna(avg_age)



avg_age2 = test.Age.mean()

test['Age'] = test.Age.fillna(avg_age2)



avg_fare = test.Fare.mean()

test['Fare'] = test.Fare.fillna(avg_fare)



train['Embarked'] = train.Embarked.fillna(method='ffill')
train.info()
test.info()
oe = preprocessing.LabelEncoder()

oe.fit(train['Sex'])

train['Sex'] = oe.transform(train['Sex'])



oe.fit(test['Sex'])

test['Sex'] = oe.transform(test['Sex'])



oe.fit(train['Embarked'])

train['Embarked'] = oe.transform(train['Embarked'])



oe.fit(test['Embarked'])

test['Embarked'] = oe.transform(test['Embarked'])
train['Age'] = train['Age'].astype('int')



test['Age'] = test['Age'].astype('int')



train['Fare'] = train['Fare'].astype('int')



test['Fare'] = test['Fare'].astype('int')
train.info()
test.info()
X = train.drop('Survived',1)

y= train['Survived']

print(X.shape)

print(y.shape)

X.head()
model = linear_model.LogisticRegression(solver ='liblinear')

model.fit(X,y)
y_predict = model.predict(test)
print(len(test))

print(len(y_predict))