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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
test.head()
train_titanic=train.drop(['Name','Ticket','Fare','SibSp','Parch','Cabin'],axis=1)

test_titanic=test.drop(['Name','Ticket','Fare','SibSp','Parch','Cabin'],axis=1)
test_titanic.head()
train_titanic.isnull().count()
train_titanic.fillna(train_titanic.mean().round(0),inplace=True)

test_titanic.fillna(train_titanic.mean().round(0),inplace=True)
train_titanic.set_index('PassengerId',inplace=True)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

leb=LabelEncoder()

test_titanic.set_index('PassengerId',inplace=True)

from sklearn.preprocessing import LabelEncoder

let=LabelEncoder()

lebt=LabelEncoder()
le.fit(train_titanic['Sex'])

leb.fit(train_titanic["Embarked"].astype(str))

let.fit(test_titanic['Sex'])

lebt.fit(test_titanic["Embarked"].astype(str))
sex=le.transform(train_titanic['Sex'])

emb=leb.transform(train_titanic['Embarked'].astype(str))

sext=le.transform(test_titanic['Sex'])

embt=leb.transform(test_titanic['Embarked'].astype(str))
train_titanic['Sex']=sex

train_titanic['Embarked']=emb

test_titanic['Sex']=sext

test_titanic['Embarked']=embt
X=train_titanic.drop(['Survived'],axis=1)

y=train_titanic['Survived']
X

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(X,y)
test_titanic.head()
predict=knn.predict(test_titanic)
data={'PassengerId':test['PassengerId'],'Survived':predict}    
df=pd.DataFrame(data)
df.to_csv('predictions.csv', index=False)