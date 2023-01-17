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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df_train = pd.read_csv('/kaggle/input/data-mining-kaggle-competition/train.csv')

X = df_train[['Pclass','SibSp', 'Parch','Sex','Fare']]
X['Sex']= pd.get_dummies(X['Sex'])
X['Pclass']= pd.get_dummies(X['Pclass'])
Y= df_train['Survived']

X['Family']=X['Parch'] + X['SibSp'] + 1

def Family(a):
    if a>4:
        return 3
    elif a<2:
        return 1
    else:
        return 2

X['Family'] = X['Family'].map(Family)
X['Family'] = pd.get_dummies(X[['Family']]) 


def Fare(b):
    if b<10:
        return 1
    elif b<20:
        return 2
    elif b<60:
        return 3
    else:
        return 4
X['Fare'] = X['Fare'].map(Fare)
X['Fare'] = pd.get_dummies(X[['Fare']])   


clf.fit(X_train,Y_train)
print(X_train.head())

print('Accuracy on training---')
y_pred_train=clf.predict(X_train)
print(accuracy_score(Y_train,y_pred_train))

print('Accuracy on test---')
y_pred_test=clf.predict(X_test)
print(accuracy_score(Y_test,y_pred_test))
df_test = pd.read_csv('/kaggle/input/data-mining-kaggle-competition/test.csv')
X_test= df_test[['Pclass','SibSp', 'Parch','Sex','Fare']]
X_test['Sex'] = pd.get_dummies(X_test['Sex'])
X_test['Pclass']= pd.get_dummies(X_test['Pclass'])

X_test['Family']=X_test['Parch'] + X_test['SibSp'] + 1

def Family(a):
    if a>4:
        return 3
    elif a<2:
        return 1
    else:
        return 2

X_test['Family'] = X_test['Family'].map(Family)
X_test['Family'] = pd.get_dummies(X_test[['Family']]) 


def Fare(b):
    if b<10:
        return 1
    elif b<20:
        return 2
    elif b<60:
        return 3
    else:
        return 4
X_test['Fare'] = X_test['Fare'].map(Fare)
X_test['Fare'] = pd.get_dummies(X_test[['Fare']])  

y_pred_test = clf.predict(X_test)
df_test['Survived'] = y_pred_test
df_test[['PassengerId', 'Survived']].to_csv('baseline.csv', index=False)

print(X_test.head())
X_0=df_train[['Age','Survived']]
X_0['Age'].fillna(X_0['Age'].median())

def Age(a):
    if a<15:
        return 1
    elif a<40:
        return 2
    elif a<60:
        return 3
    else:
        return 4
X_0['Age'] = X_0['Age'].map(Age)
X_0['Age'] = pd.get_dummies(X_0[['Age']])                                    
print(X_0[['Age','Survived']].groupby(['Age','Survived']).size())

X_1=df_train[['Age','Name','Survived']]
X_1['Name']
X_2=df_train[['Cabin','Survived']]
X_2['Cabin']=X_2['Cabin'].fillna('X')
X_2['Cabin']=X_2['Cabin'].str[0] 
def Cabin(c):
    if c=='A'or c=='G':
        return 1
    elif c=='D' or c=='E':
        return 2
    elif c=='C'or c=='F':
        return 3
    elif c=='B':
        return 4
    else:
        return 5
X_2['Cabin'] = X_2['Cabin'].map(Cabin)
print(X_2[['Cabin','Survived']].groupby(['Cabin','Survived']).size())