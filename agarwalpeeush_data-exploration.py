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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.info()
columns_interested = ['Pclass', 'Sex','SibSp', 'Parch', 'Embarked']
for col in columns_interested:
    print(col)
    total_counts = train[col].value_counts().rename_axis(col).reset_index(name='Total')
    survived_counts = train[train['Survived'] == 1][col].value_counts().rename_axis(col).reset_index(name='Survived')
    temp = pd.merge(survived_counts, total_counts, on=col)
    temp['Survival%'] = 100* temp['Survived']/temp['Total']
    print(temp.sort_values(by='Survival%', ascending=False))
    print('-'*10)
train['Fare'].hist()
train[train['Survived'] == 1]['Fare'].hist()
# Name, Pclass of top 5 oldest people
# Method 1:

train.sort_values(by='Age', ascending=False)[['Age','Name', 'Pclass']].head()
# Name, Pclass of top 5 oldest people
# Method 2: 

train.nlargest(5, 'Age').filter(items=['Age', 'Name', 'Pclass'])
# Name, Pclass of top 5 youngest people
# Method 1:

train.sort_values(by='Age')[['Age','Name', 'Pclass']].head()
# Name, Pclass of top 5 youngest people
# Method 2:

train.nsmallest(5, 'Age')[['Age','Name', 'Pclass']]
# People travelling solo
# Method 1

train.loc[:, 'IsAlone'] = train[['SibSp', 'Parch']].apply(lambda cols:(cols[0], cols[1]) == (0, 0), axis=1)
train['IsAlone'] = train['IsAlone'].astype('int')
train['IsAlone'].value_counts()
len(train[train['IsAlone']==1])
# People travelling solo
# Method 2

train['Alone'] = (train['SibSp']==0) & (train['Parch'] == 0)
train['Alone'].value_counts()
len(train[train['Alone']==1])
# People travelling solo
# Method 3

train.query('SibSp == 0 and Parch == 0')['Survived'].value_counts()
# Survival rate of people travelling solo
100* len(train[(train['Survived']==1) & (train['Alone']==1)]) / 537
# How many paid > $200
len(train[train['Fare']>200])
# How many paid > $200 and survived
100* len(train[(train['Survived']==1) & (train['Fare']>200)]) / 20
# Survival rate of Pclass=1
total_pclass_1 = len(train[train['Pclass']==1])
100 * len(train[(train['Survived']==1) & (train['Pclass']==1)]) / total_pclass_1
# Survival rate of females
total_females = len(train[train['Sex']=='female'])
100 * len(train[(train['Survived']==1) & (train['Sex']=='female')]) / total_females
