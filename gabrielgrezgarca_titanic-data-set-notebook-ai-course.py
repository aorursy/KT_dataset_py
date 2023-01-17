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
with open('../input/titanic/train.csv', 'r', encoding='utf-8') as f:
    train = pd.read_csv(f)
    train.name = 'Train'
    
with open('../input/titanic/test.csv', 'r', encoding='utf-8') as f:
    test = pd.read_csv(f)
    test.name = 'Test'
    
train.head(10) # train set
test.head(10) # test set
train.info()
train.describe(include='all')
train[['Pclass', 'Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False) # with Pclass
train[['Sex', 'Survived']].groupby('Sex').mean().sort_values(by='Survived', ascending=False) # with gender
train[['Parch', 'Survived']].groupby('Parch').mean().sort_values(by='Survived', ascending=False) # with parents or children
train[['SibSp', 'Survived']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False) # with siblings or spouses


train[['Embarked', 'Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False) # with embarked location


import seaborn as sb
sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sb.factorplot(x='Survived',col='Sex',kind='count',data=train)
sb.countplot(x='Survived',hue='Pclass',data=train)
sb.countplot(x='SibSp',data=train)
sb.countplot(x='Age',data=train)
sb.boxplot(x='Pclass',y='Age',data=train,palette='winter')
sb.violinplot(x='Sex',y='Age',hue='Survived',data=train,split=True)
def impute(cols):
    Age = cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute,axis=1)
train.drop('Cabin',axis=1,inplace=True)
sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
