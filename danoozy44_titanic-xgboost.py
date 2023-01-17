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

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train
train = train.drop(columns= ['Name','Ticket','Cabin'])

test = test.drop(columns= ['Name','Ticket','Cabin'])
train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)

train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)

train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

train['Gender'] = (train['Sex'] == 'male').astype(int)
test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)

test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)

test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)

test['Gender'] = (test['Sex'] == 'male').astype(int)
train = train.drop(columns = ['Sex'])

test = test.drop(columns = ['Sex'])
train = train.drop(columns = ['Embarked'])

test = test.drop(columns = ['Embarked'])
train.fillna(0, inplace=True)

test.fillna(0, inplace=True)
X = train.drop(columns = ['Survived'])

y = train['Survived']
from xgboost import XGBClassifier



model = XGBClassifier(learning_rate=0.01, n_estimators=1000)

model.fit(X,y)
pred = model.predict(test)
result = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred})



result.to_csv("submission.csv", index=False)