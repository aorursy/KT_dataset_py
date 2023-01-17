# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC



from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv", dtype={'Age': np.float64})

result = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

full_data = [train, test]

for data in full_data:

  

  data['Sex'] = data['Sex'].fillna(0)

  data['Sex'] = data['Sex'].map( {'female': 0 , 'male': 1  })







  embarked_map = { "S" : 0, "C": 1  , "Q" : 2}

  data['Embarked'] = data['Embarked'].map(embarked_map)

  data['Embarked'] = data['Embarked'].fillna(0)





  data['Fare'] = data['Fare'].fillna(0)

  data.loc[train['Fare'] <= 7.91 , 'Fare'] = 0

  data.loc[(train['Fare'] > 7.91) & (data['Fare'] <=14.454) , 'Fare'] = 1

  data.loc[(train['Fare'] > 14.454) & (data['Fare'] <=31) , 'Fare'] = 2

  data.loc[train['Fare'] > 31 ,'Fire'] = 3

  data['Fare'].astype(float)







  data['Age'] = train['Age'].fillna(0)

  data.loc[train['Age'] <= 16 , 'Age'] = 0

  data.loc[(train['Age'] > 16) & (data['Age'] <=32) , 'Age'] = 1

  data.loc[(train['Age'] > 32) & (data['Age'] <=48) , 'Age'] = 2

  data.loc[(train['Age'] > 48) & (data['Age'] <=64) , 'Age'] = 2

  data.loc[train['Age'] > 64 , 'Age'] = 3

  data['Age'] = data['Age'].astype(int)





  

drop_elements  = ['PassengerId' ,'Name' , 'Ticket' , 'Cabin', 'SibSp' , 'Parch' ,'Fire' ]

train = train.drop(drop_elements, axis=1)

test = test.drop(drop_elements, axis=1)

result = result.drop(['PassengerId'], axis=1)







x = train.drop(['Survived'] , axis=1)

y = train['Survived']

sgd = SVC()

sgd.fit(x,y)

pred = sgd.predict(test)

acc = accuracy_score(result, pred)

print(acc)

#0.715311004784689
test1 = pd.read_csv("/kaggle/input/titanic/test.csv", dtype={'Age': np.float64})

holdout_ids = test1["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": pred}

submission = pd.DataFrame(submission_df)
submission.to_csv("submission.csv",index=False)