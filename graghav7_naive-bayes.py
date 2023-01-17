
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

#train_df = train_df.drop(labels = ['PassengerId','Name','Ticket','Cabin'],axis=1)
#Only 3 variables effectively used. NO DISTRIBUTIONS USED

train_df = train_df.drop(labels = ['PassengerId','Name','Ticket','Cabin','Age','Fare','Parch','SibSp'],axis=1)

train_df = train_df.fillna(train_df.mean())

train_df = train_df.replace('female',0)
train_df = train_df.replace('male',1)
train_df = train_df.replace('C',0)
train_df = train_df.replace('Q',1)
train_df = train_df.replace('S',2)

train_df1 = train_df[train_df['Survived'] == 1]
train_df0 = train_df[train_df['Survived']==0]

Pclass_df1 = train_df1.groupby('Pclass').count()
Sex_df1 = train_df1.groupby('Sex').count()
#Age_df1 = train_df1.groupby('Age').count()
#SibSp_df1 = train_df1.groupby('SibSp').count()
#Parch_df1 = train_df1.groupby('Parch').count()
#Fare_df1 = train_df1.groupby('Fare').count()
Embarked_df1 = train_df1.groupby('Embarked').count()

Pclass_df0 = train_df0.groupby('Pclass').count()
Sex_df0 = train_df0.groupby('Sex').count()
#Age_df0 = train_df0.groupby('Age').count()
#SibSp_df0 = train_df0.groupby('SibSp').count()
#Parch_df0 = train_df0.groupby('Parch').count()
#Fare_df0 = train_df0.groupby('Fare').count()
Embarked_df0 = train_df0.groupby('Embarked').count()

Pclass1 = np.array(Pclass_df1['Survived'])
Sex1 = np.array(Sex_df1['Survived'])
#Age1 = np.array(Age_df1['Survived'])
#SibSp1 = np.array(SibSp_df1['Survived'])
#Parch1 = np.array(Parch_df1['Survived'])
Embarked1 = np.array(Embarked_df1['Survived'])

Pclass0 = np.array(Pclass_df0['Survived'])
Sex0 = np.array(Sex_df0['Survived'])
#Age0 = np.array(Age_df0['Survived'])
#SibSp0 = np.array(SibSp_df0['Survived'])
#Parch0 = np.array(Parch_df0['Survived'])
Embarked0 = np.array(Embarked_df0['Survived'])
Pclass1 = Pclass1/sum(Pclass1)
Sex1 = Sex1/sum(Sex1)
#Age1 = Age1/sum(Age1)
#SibSp1 = SibSp1/sum(SibSp1)
#Parch1 = Parch1/sum(Parch1)
Embarked1 = Embarked1/sum(Embarked1)
                             
Pclass0 = Pclass0/sum(Pclass0)
Sex0 = Sex0/sum(Sex0)
#Age0 = Age0/sum(Age0)
#SibSp0 = SibSp0/sum(SibSp0)
#Parch0 = Parch0/sum(Parch0)
Embarked0 = Embarked0/sum(Embarked0)
test_df = pd.read_csv('../input/test.csv')

test_df = test_df.drop(labels = ['PassengerId','Name','Ticket','Cabin','Age','Fare','Parch','SibSp'],axis=1)

#test_df = test_df.fillna(train_df.mean())

test_df = test_df.replace('female',0)
test_df = test_df.replace('male',1)
test_df = test_df.replace('C',0)
test_df = test_df.replace('Q',1)
test_df = test_df.replace('S',2)


X_data = np.matrix(test_df)
X_prob1 = np.matrix(test_df)
X_prob0 = np.matrix(test_df)

X_prob1 = X_prob1.astype(float)
X_prob0 = X_prob0.astype(float)
X_prob1[:,0] = Pclass1[X_data[:,0]-1]
X_prob1[:,1] = Sex1[X_data[:,1]]
#X_prob1[:,2] = SibSp1[X_data[:,2]]
#X_prob1[:,3] = Parch[X_data[:,3]]
X_prob1[:,2] = Embarked1[X_data[:,2]]

X_prob0[:,0] = Pclass0[X_data[:,0]-1]
X_prob0[:,1] = Sex0[X_data[:,1]]
#X_prob0[:,2] = SibSp0[X_data[:,2]]
#X_prob0[:,3] = Parch[X_data[:,3]]
X_prob0[:,2] = Embarked0[X_data[:,2]]

y_pred1 = np.multiply(X_prob1[:,0],X_prob1[:,1],X_prob1[:,2])[:,0]
y_pred0 = np.multiply(X_prob0[:,0],X_prob0[:,1],X_prob0[:,2])[:,0]

y = (y_pred1>y_pred0)
y_pred = y.astype(int)