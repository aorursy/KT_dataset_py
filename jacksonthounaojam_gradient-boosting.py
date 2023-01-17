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
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

test_set = pd.read_csv('/kaggle/input/titanic/test.csv')



train_set.head()

#test_set.shape
from sklearn.ensemble import GradientBoostingClassifier as gb

from sklearn.model_selection import train_test_split





train_set['Sex'] = pd.get_dummies(train_set['Sex']).fillna('Female')

train_set['Cabin'] = pd.get_dummies(train_set['Cabin'])

train_set['Cabin'] = train_set['Cabin'].fillna(train_set.Cabin.mean())

train_set['Fare'] = train_set['Fare'].fillna(train_set.Fare.mean())

train_set['Age'] = -1*(train_set['Age'].fillna(train_set['Age'].mean()))

train_set['Pclass'] = -1*((train_set['Pclass']).fillna(train_set['Pclass'].mean()))

test_set['Sex'] = pd.get_dummies(test_set['Sex']).fillna('Female')

test_set['Cabin'] = pd.get_dummies(test_set['Cabin'])

test_set['Cabin'] = test_set['Cabin'].fillna(test_set['Cabin'].mean())

test_set['Pclass'] = -1*((test_set['Pclass']).fillna(test_set['Pclass'].mean()))

test_set['Age'] = -1*(test_set['Age'].fillna(test_set['Age'].mean()))

test_set['Fare'] = test_set['Fare'].fillna(test_set.Fare.mean())



features = ['Pclass','Age','Fare','Sex','Cabin']



X = train_set[features]

y = train_set["Survived"]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,random_state=23)







test = test_set[features]

gradient=gb(learning_rate = 0.01, max_depth = 12, random_state = 0).fit(X_train , y_train)

#predict = gradient.predict(X_valid)

predict = gradient.predict(test)





#print("a :",gradient.score(X_train,y_train))

#print("b :",gradient.score(X_valid,y_valid))



output = pd.DataFrame({'PassengerId':test_set.PassengerId,'Survived':predict})



output.to_csv('gradient.csv',index=False)

print('submission successful')


