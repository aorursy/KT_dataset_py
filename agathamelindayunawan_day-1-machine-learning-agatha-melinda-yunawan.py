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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head(5)
df.isnull().sum(axis=1)
(df.isnull().sum(axis=1) > 0).sum()
df.groupby('Sex').size()
df.groupby('Sex').size() / len(df) * 100
df['Cabin'].nunique()
df['Cabin'].unique()
df[df['Parch'] >= 2]
len(df[df['Parch'] >= 2])
df[(df['SibSp'] + df['Parch']) == 0]
df[(df['SibSp'] + df['Parch']) == 0].groupby('Pclass').size()
df[(df['SibSp'] + df['Parch']) == 0].groupby('Pclass').size().sort_values(ascending=False).head(1)
df.groupby('Embarked')['Fare'].mean().sort_values(ascending=False).head(1)
df[df['Age'] >= 50]
df[df['Age'] >= 50].groupby('Cabin').size().sort_values(ascending=False).head(1)
df.head(5)
df['Survived'].mean()
df[['Pclass', 'Survived']].groupby(['Pclass']).agg(['count', 'sum', 'mean'])
df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).agg(['count', 'sum', 'mean'])
df[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex']).agg(['count', 'sum', 'mean'])
df[['Embarked', 'Survived']].groupby(['Embarked']).agg(['count', 'sum', 'mean'])
df[['Pclass', 'Embarked', 'Sex', 'Survived']].groupby(['Pclass', 'Embarked', 'Sex']).agg(['count', 'sum', 'mean'])