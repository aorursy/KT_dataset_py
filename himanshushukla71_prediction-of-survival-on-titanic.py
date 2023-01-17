# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")
data_test=pd.read_csv("../input/test.csv")
data.head(2)

data_test.head()
data.info()
col_delete=['PassengerId','Name','Ticket','Cabin','Embarked']
data.drop(col_delete,axis=1,inplace=True)
data_test.drop(col_delete,axis=1,inplace=True)
data.head()
data_test.head()
data.isnull().sum()


data_test.isnull().sum()
data.Age.fillna(data.Age.mean(),inplace=True)
data_test.Age.fillna(data_test.Age.mean(),inplace=True)
data_test.Fare.fillna(data_test.Fare.mean(),inplace=True)
def f(s):
    if s=='male':
        return 0
    else:
        return 1
data['Sex']=data.Sex.apply(f)
data_test['Sex']=data_test.Sex.apply(f)

data.head()
x_variable=['Pclass','Sex','Age','Parch','Fare']
train_x=data[x_variable]
train_y=data.Survived
test_x=data_test[x_variable]
model=RandomForestClassifier()
model.fit(train_x,train_y)
model.score(train_x,train_y)
result=model.predict(test_x)
df=pd.DataFrame(result)
df.head()
test=pd.read_csv("../input/test.csv")
df['PassengerId']=test['PassengerId']
df.head()
df.columns = ["Survived", "PassengerId"]
df.head()
df.to_csv('submission.csv', index=False)