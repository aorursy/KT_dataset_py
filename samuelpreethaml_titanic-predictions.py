# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import sklearn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/train.csv')
data.Sex=data['Sex'].replace({'male':1,'female':0})
data.Embarked=data['Embarked'].replace({'S':0,'C':1,'Q':2})
X=data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
X=X.fillna(X.mean())
y=data.Survived
m=RandomForestClassifier(random_state=1)
m.fit(X,y)
t=d=pd.read_csv('../input/test.csv')
t.Sex=t['Sex'].replace({'male':1,'female':0})
t.Embarked=t['Embarked'].replace({'S':0,'C':1,'Q':2})
t=t.fillna(t.mean())
p=t[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
h=pd.DataFrame(t.PassengerId)
h['Survived']=m.predict(p)
h.to_csv('take.csv',index=False)
# Any results you write to the current directory are saved as output.