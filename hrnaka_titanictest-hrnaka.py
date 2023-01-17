#kaggle test



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



df = pd.read_csv('/kaggle/input/titanic/train.csv')

testdf = pd.read_csv('/kaggle/input/titanic/test.csv')



df.isnull().sum()



from sklearn.model_selection import train_test_split



df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Age'] = df['Fare'].fillna(df['Fare'].median())

df['Embarked'] = df['Embarked'].fillna('S')



df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

df['Embarked'] = df['Embarked'].map( {'S': 0 , 'C':1 , 'Q':2}).astype(int)

df = df.drop(['Cabin','Name','PassengerId','Ticket'],axis =1)



train_X = df.drop('Survived',axis = 1)

train_y = df.Survived

(train_X , test_X , train_y , test_y) = train_test_split(train_X, train_y , test_size = 0.3, random_state = 0)



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state = 0)

clf = clf.fit(train_X , train_y)

pred = clf.predict(test_X)



#正解率の算出

from sklearn.metrics import (roc_curve , auc ,accuracy_score)

pred = clf.predict(test_X)

fpr, tpr, thresholds = roc_curve(test_y , pred,pos_label = 1)

auc(fpr,tpr)

accuracy_score(pred,test_y)