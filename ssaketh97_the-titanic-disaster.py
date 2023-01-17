# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

#input the data

train = pd.read_csv("../input/train.csv");

test = pd.read_csv('../input/test.csv');

test_new = pd.read_csv('../input/test.csv');
train.head()
train.info()
train[train['Embarked'].isnull()]
train['Embarked'].fillna('C',inplace=True);

train[train['Embarked'].isnull()]
train[train['Age'].isnull()]
print('Mr is for people between')

print(train[train['Name'].str.contains("Mr\.")]['Age'].min())

print('and')

print(train[train['Name'].str.contains("Mr\.")]['Age'].max())

print('Mean is')

print(train[train['Name'].str.contains("Mr\.")]['Age'].mean())

print(' ')



print('Master is for people between')

print(train[train['Name'].str.contains("Master\.")]['Age'].min())

print('and')

print(train[train['Name'].str.contains("Master\.")]['Age'].max())

print('Mean is')

print(train[train['Name'].str.contains("Master\.")]['Age'].mean())

print(' ')



print('Mrs is for people between')

print(train[train['Name'].str.contains("Mrs\.")]['Age'].min())

print('and')

print(train[train['Name'].str.contains("Mrs\.")]['Age'].max())

print('Mean is')

print(train[train['Name'].str.contains("Mrs\.")]['Age'].mean())

print(' ')



print('Miss is for people between')

print(train[train['Name'].str.contains("Miss\.")]['Age'].min())

print('and')

print(train[train['Name'].str.contains("Miss\.")]['Age'].max())

print('Mean is')

print(train[train['Name'].str.contains("Miss\.")]['Age'].mean())

print(' ')



x = train[train['Age'].isnull()].index.tolist()

for i in x:

    name = train.iloc[i]['Name']

    if(str('Mr.') in name):

        if(train.iloc[i]['Parch']>0):

                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(11, 16)

        else:

                 train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(16, 50)

    elif(str('Master.') in name):

          train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(1, 10)

    elif(str('Mrs.') in name):

         train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(21, 63)

    elif(str('Miss.') in name):

        if(train.iloc[i]['Parch']>0):

                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(1, 18)

        else:

                train.loc[train['PassengerId']==(i+1),'Age'] = random.randrange(18, 35)



train[train['Age'].isnull()]
train.loc[train['PassengerId']==767,'Age']=40

train[train['Age'].isnull()]
train.info()
train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)

train.info()
train=pd.get_dummies(train,columns=['Sex','Embarked'])

train.drop(['Sex_female','Embarked_Q'],axis=1,inplace=True)

train.head()
plt.figure(figsize=(12,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train_x = train.iloc[:,1:]

train_y= train.iloc[:,0]



train_x = preprocessing.scale(train_x)

train_x
classifier=Sequential()

classifier.add(Dense(output_dim=12,init='uniform',activation='relu',input_dim=8))

classifier.add(Dropout(0.3))

classifier.add(Dense(output_dim=8,init='uniform',activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(train_x,train_y,batch_size=10,epochs=300)
test.head()

passengers = test['PassengerId'] 

print(test.info())
# handle missing age

y = test[test['Age'].isnull()].index.tolist()

for i in y:

    name = test.iloc[i]['Name']

    if(str('Mr.') in name):

        if(test.iloc[i]['Parch']>0):

                test.loc[i,'Age'] = random.randrange(11, 16)

        else:

                 test.loc[i,'Age'] = random.randrange(16, 50)

    elif(str('Master.') in name):

          test.loc[i,'Age'] = random.randrange(1, 10)

    elif(str('Mrs.') in name):

         test.loc[i,'Age'] = random.randrange(21, 63)

    elif(str('Miss.') in name):

        if(test.iloc[i]['Parch']>0):

                test.loc[i,'Age'] = random.randrange(1, 18)

        else:

                test.loc[i,'Age'] = random.randrange(18, 35)



test.loc[88,'Age']= random.randrange(18, 35)               

test[test['Age'].isnull()]

#handle missing fare

test.loc[152,'Fare']=7.7500

test[test['Fare'].isnull()]


test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)

test.info()
test=pd.get_dummies(test,columns=['Sex','Embarked'])

test.drop(['Sex_female','Embarked_Q'],axis=1,inplace=True)



test = preprocessing.scale(test)

test
y_pred = classifier.predict(test) 

pred=[]

for i in range(0,y_pred.shape[0]):

    if(y_pred[i]>0.5):

        pred.append(1)

    else:

        pred.append(0)
test_new.head()
output = pd.DataFrame({'PassengerId': test_new.PassengerId,'Survived': pred})

output.to_csv('submission.csv', index=False)