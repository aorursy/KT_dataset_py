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
train= pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
train.head()


gender = {'male': 0,'female': 1} 

train.Sex = [gender[item] for item in train.Sex]

test.Sex = [gender[item] for item in test.Sex]

  
train.head()
train.dropna(axis=1,inplace=True)

train.Fare=train.Fare/32.0

test.Fare=test.Fare/32.0

test.head()
train.corr()
from sklearn.model_selection import train_test_split

Y=train.Survived

X=train[['Pclass','Sex','SibSp','Parch','Fare']]



X_train, X_test, y_train, y_test = train_test_split(

    X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction)
test.fillna(test.mean(),inplace=True)

test.head()
predictions=clf.predict(test[['Pclass','Sex','SibSp','Parch','Fare']])

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

submission
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)