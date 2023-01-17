# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier

a=pd.read_csv('../input/train.csv')
b=a.drop(['Pclass','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Name'],axis=1)
b['Age']=b['Age'].fillna(b['Age'].mean())
k=KNeighborsClassifier(n_neighbors=5)
b1=b.Survived
b=b.drop(['PassengerId','Survived'],axis=1)
b['Sex']=b.Sex.replace(['male','female'],[1,0])
s=k.fit(b,b1)
d1=pd.read_csv('../input/test.csv')

d=pd.read_csv('../input/test.csv')
d=d[b.columns]
d.Sex=d.Sex.replace(['male','female'],[1,0])

d.Age=d.Age.fillna(d.Age.mean())
result=s.predict(d)
f=pd.DataFrame(result)
f.index=d1.PassengerId
f.columns=['Survived']

f.index.name='PassengerId'
f.to_csv('result.csv')