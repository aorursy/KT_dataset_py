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
import pandas as pd
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.describe()
# help(train)
train.dtypes
train.iloc[0:100,0:5]
train.loc[0:100,'Survived']
train
train.dropna()
print(len(train['Cabin']) - train['Cabin'].count())

print(len(train['Embarked']) - train['Embarked'].count())
train  =  train.drop(['Cabin'],axis = 1)
train  =  train.drop(['PassengerId'],axis = 1)
train  =  train.drop(['Name'],axis = 1)
train  =  train.drop(['Ticket'],axis = 1)
train
print('Survived : ' + str( len(train['Survived']) - train['Survived'].count()) )

print('Pclass : ' + str( len(train['Pclass']) - train['Pclass'].count()))

print('Sex : ' + str( len(train['Sex']) - train['Sex'].count()))

print('Age : ' + str( len(train['Age']) - train['Age'].count()))

print('SibSp : ' + str( len(train['SibSp']) - train['SibSp'].count()))

print('Parch : ' + str( len(train['Parch']) - train['Parch'].count()))

print('Ticket : ' + str( len(train['Ticket']) - train['Ticket'].count()))

print('Fare : ' + str( len(train['Fare']) - train['Fare'].count()))

print('Embarked : ' + str( len(train['Embarked']) - train['Embarked'].count()))
print(train['Age'].mean())
train['Age'] = train['Age'].fillna(train['Age'].mean())
print('Age : ' + str( len(train['Age']) - train['Age'].count()))

train = train.dropna()
train.dtypes
set(train['Sex'])
Sex_data_dic = {'female':0,'male':1}
train['Sex'] = train['Sex'].map(Sex_data_dic)
train.dtypes
set(train['Sex'])
set(train['Embarked'])
Embarked_data_dic = {'C':0, 'Q':1, 'S':2}
train['Embarked'] = train['Embarked'].map(Embarked_data_dic)
set(train['Embarked'])
train.describe()
train.head(10)