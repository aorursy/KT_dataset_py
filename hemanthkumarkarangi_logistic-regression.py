# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_test  = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_test.sample()
x=titanic_train.copy()
y=titanic_train['Survived']
y.columns=[0]
x_t=titanic_test.copy()
x_t.info()
y=pd.DataFrame(y)
x=x.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)
x_t=x_t.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
print(x.head())
x_t.head()
ones=np.ones((891,1))
ones=pd.DataFrame(data=ones,columns=['ones'])
x=pd.merge(ones,x,left_index=True,right_index=True) #creating bias column of ones.
x.head()
ones_t=np.ones((418,1))
ones_t=pd.DataFrame(data=ones_t,columns=['ones'])
x_t=pd.merge(ones_t,x_t,left_index=True,right_index=True) #creating bias column of ones.
x_t.head()
x['Sex']=x['Sex'].replace(['male','female'],[1,0])
x['Embarked']=x['Embarked'].replace(['C','Q','S'],[1,2,3])
x['Embarked']=x['Embarked'].replace(np.NaN,2)
mean_age=round(np.mean(x['Age']))
x['Age']=x['Age'].replace(np.NaN,mean_age)
m=len(x)

x_t['Sex']=x_t['Sex'].replace(['male','female'],[1,0])
x_t['Embarked']=x_t['Embarked'].replace(['C','Q','S'],[1,2,3])
x_t['Embarked']=x_t['Embarked'].replace(np.NaN,2)
x_t['Fare']=x_t['Fare'].replace(np.NaN,12.459678)
mean_age=round(np.mean(x_t['Age']))
x_t['Age']=x_t['Age'].replace(np.NaN,mean_age)
m_t=len(x_t)
#print(x.head())
#print(x_t.head())
x_t['Fare'].isnull().values.any()

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model=LogisticRegression(C=1E6)
model=model.fit(x,y)
prediction=model.predict(x_t)
output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")