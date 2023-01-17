# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

train =pd.read_csv("../input/titanic/train.csv")

os.listdir("../input") 

# Any results you write to the current directory are saved as output.
train.head()
train['Sex']=train['Sex'].map({'male':0,'female':1})

train['Embarked']=train['Embarked'].map({'S':0,'C':1})

train =train[['Pclass','Sex','Age','Survived','Embarked']]
train.head()
train =train.dropna()
test =pd.read_csv("../input/titanic/test.csv")

test.head()
test['Sex']=test['Sex'].map({'male':0,'female':1})

test['Embarked']=test['Embarked'].map({'S':0,'C':1})

test =test[['Pclass','Sex','Age','Embarked']]
trainy = train['Survived']

trainx =train.drop('Survived',axis=1)
model =XGBClassifier()

model.fit(trainx,trainy)
ypred =model.predict(test)

predictions =[round(value) for value in ypred]
print(predictions)
pd.read_csv("../input/titanic/gender_submission.csv")
#newdf =pd.DataFrame(predictions)

#newdf.head()
b =pd.read_csv("../input/titanic/test.csv")



submission = pd.DataFrame({

        "PassengerId": b["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('submission.csv', index=False)