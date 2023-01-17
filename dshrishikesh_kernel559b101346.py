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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gs=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()
test.head()
ta=pd.concat([gs,test.drop(['PassengerId'],1)],1)

ta.head()
train.drop(train.index[473:891],inplace=True)

train.tail(5)
train=pd.concat([train,ta],0)

train.tail(5)
train=train.drop(['Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis='columns')

train.tail(2)
test=test.drop(['Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis='columns')

test.head(2)
dummies_train=pd.get_dummies(train[['Sex']])

dummies_train.head(2)

dummies_test=pd.get_dummies(test[['Sex']])

dummies_test.head(2)
train=train.drop(['Sex'],axis='columns')

test=test.drop(['Sex'],axis='columns')
train=pd.concat([train,dummies_train.drop(['Sex_male'],axis='columns')],axis='columns')

train.head(2)
test=pd.concat([test,dummies_test.drop(['Sex_male'],axis='columns')],axis='columns')

test.head(2)
train.isnull().sum()
train.fillna(train.median(),inplace=True)
test.isnull().sum()
test.fillna(train.median(),inplace=True)
from sklearn.naive_bayes import GaussianNB,MultinomialNB

GNB=GaussianNB()

x=train.drop(['Survived','PassengerId'],axis='columns')

x.head()
y=train['Survived']

y.head(2)
GNB.fit(x,y)



predictions = GNB.predict(test.drop(['PassengerId'],axis='columns'))



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")