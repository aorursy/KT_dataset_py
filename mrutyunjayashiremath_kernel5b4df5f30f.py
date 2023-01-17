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
#This titanic dataset using DecisionTreeClassifier algorithm 
#now uploading pandas library  and titanic dataset

import pandas as pd

df = pd.read_csv("../input/titanicdataset-traincsv/train.csv")

df.head()
#drop uneccesary feature columns  or clecaning dataset

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)

df.head()
# now drop the Survived columns 

inputs = df.drop('Survived',axis='columns')

target = df.Survived
#now this is converting string to interger

inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
inputs.Age[:10]
#filling nan 

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()
#splitin data

from sklearn.model_selection import train_test_split
#spliting data

X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
len(X_train)
len(X_train)
len(X_test)
len(X_train)
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.score(X_train,y_train)