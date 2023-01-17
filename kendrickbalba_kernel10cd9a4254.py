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



train = pd.read_csv('../input/titanic/train.csv')

train.head(3)
import pandas as pd



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train = train.drop(['Name','Ticket', 'Fare', 'Cabin'],axis=1)

test = test.drop(['Name','Ticket', 'Fare', 'Cabin'],axis=1)



for df in [train,test]:

    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})



for df in [train,test]:

    df['Emba']=df['Embarked'].map({'C':1,'Q':2, 'S':3})

    

train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)

train['Emba'] = train['Emba'].fillna(0)

test['Emba'] = test['Emba'].fillna(0)



features = ['Pclass','Age','Sex_binary', 'SibSp', 'Parch', 'Emba']

target = 'Survived'



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5,min_samples_leaf=2)

clf.fit(train[features],train[target]) 



predictions = clf.predict(test[features])



submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)