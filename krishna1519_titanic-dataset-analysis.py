
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
import matplotlib.pyplot as plt
import seaborn as s
s.set()
        
train = pd.read_csv('/kaggle/input/titanicdataset-traincsv/train.csv')
print(train.head())

#print(train.isnull().sum())
#print(train.info())
#print(train.describe())
#s.heatmap(train.isnull(), yticklabels=False)

# analysis on survived column
s.set_style('whitegrid')
s.countplot(x='Survived', data=train)
plt.show()

s.set_style('whitegrid')
s.countplot(x='Survived', data=train, hue = 'Sex')
plt.show()

s.set_style('whitegrid')
s.countplot(x='Survived', data=train, hue = 'Pclass')
plt.show()

s.distplot(train['Age'].dropna(), kde=False,color='darkred', bins=40)

# filling NA values for age
train['Age']=train['Age'].fillna(train['Age'].mean())
train.drop('Cabin', axis =1 , inplace=True)

s.heatmap(train.isnull(), yticklabels=False)

train.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
sex=pd.get_dummies(train['Sex'], drop_first=True)
embark=pd.get_dummies(train['Embarked'], drop_first=True)

train_final=pd.concat([train,sex,embark],axis =1)

train.drop(['Sex','Embarked'], axis =1, inplace=True)
print(train_final.head())





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
