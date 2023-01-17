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
import pandas as pd

import numpy as np
df=pd.read_csv("../input/titanic-subset/Titanic.csv")

df
df.drop(["PassengerId","Name","SibSp","Ticket","Cabin","Embarked"],axis="columns")
inputs=df.drop('Survived',axis='columns')
target=df['Survived']
from sklearn.preprocessing import LabelEncoder

le_Sex=LabelEncoder()
inputs['Sex_n']=le_Sex.fit_transform(inputs['Sex'])
inputs
inputs_n=inputs.drop(['Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','PassengerId'],axis='columns')
inputs_n.Age = inputs_n.Age.fillna(inputs_n.Age.mean())
inputs_n
target
from sklearn import tree

model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
model.score(inputs_n,target)