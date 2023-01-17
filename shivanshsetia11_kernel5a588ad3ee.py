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
data = pd.read_csv('/kaggle/input/titanic/train.csv')
print(data.head())

col = data.columns
data.shape
#extra = pd.isnull(data)

#b = pd.isnull(data['Cabin'])

#print(data[b])

for i in col:

    bools = pd.isnull(data[i])

    print(i + " " + str(data[bools].shape))
df = data.drop(['Cabin','Age','Name','PassengerId'],axis=1)
df.shape
df = df.dropna()
df.shape
df.head()
df['Sex'].replace({"male":0,"female":1},inplace=True)
df.head()
df['Embarked'].unique()
df["Embarked"].replace({'S':0,"C":1,"Q":2},inplace=True)
df.head()
df = df.drop(['Ticket'],axis=1)
X = df.drop(['Survived'],axis=1)

y = df['Survived']
from sklearn.ensemble import RandomForestClassifier

  

 # create regressor object 

clf = RandomForestClassifier(n_estimators = 100, random_state = 0) 

  

# fit the regressor with X and y data 

clf.fit(X, y)   
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.shape

test.columns
test_X = test.drop(['Cabin','Age','Name','Ticket'],axis=1)

test_X["Embarked"].replace({'S':0,"C":1,"Q":2},inplace=True)

test_X['Sex'].replace({"male":0,"female":1},inplace=True)
test_X.dropna(inplace=True)

test_X.reset_index(drop=True, inplace=True)

passenger = test_X['PassengerId']

test_X = test_X.drop(['PassengerId'],axis=1)

print(test_X.shape)

p=[]

for i in clf.predict(test_X):

    p.append(i)
passenger.shape
#pred = {'Survived':clf.predict(test_x)}

ds=[]

for i in range(len(p)):

    ds.append( [passenger[i],p[i]]                    )

final = pd.DataFrame(ds, columns = ['PassengerId', 'Survived'])
final.head()