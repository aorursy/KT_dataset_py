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
train=pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')

test=pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')
train.Parch.value_counts()
#Removing unnecessary column

train.drop(['Name','Ticket','Cabin','Parch'],inplace=True,axis=1)

test.drop(['Name','Ticket','Cabin','Parch'],inplace=True,axis=1)
#Checking for null values

train.isnull().sum()
test.shape
#Removing null values  with mean

train.Age.fillna(train.Age.mean(),inplace=True)

test.Age.fillna(test.Age.mean(),inplace=True)

train.Embarked.fillna('S',inplace=True)

test.Fare.fillna(test.Fare.mean(),inplace=True)
test.shape
test.isnull().sum()
#Converting categorical values

train=pd.get_dummies(train,columns=['Pclass','Embarked'],drop_first=True)

test=pd.get_dummies(test,columns=['Pclass','Embarked'],drop_first=True)

from sklearn.preprocessing import LabelEncoder

encode=LabelEncoder()

train['Sex']=encode.fit_transform(train.Sex)

test['Sex']=encode.fit_transform(test.Sex)
test
#SVM classifier

from sklearn.svm import SVC

model=SVC(kernel='linear')

X=train.drop('Survived',axis=1)

y=train['Survived']
#Splitting the dataset into train and valid set

from sklearn.model_selection import train_test_split 

xtrain,xvalid,ytrain,yvalid=train_test_split(X,y,test_size=0.25)
model.fit(xtrain,ytrain)
model.score(X,y)
y=model.predict(test)
sub=test.index
sub=pd.DataFrame(test.index)
sub['Survived']=model.predict(test)
sub=sub.set_index('PassengerId')
sub.to_csv('submission.csv')