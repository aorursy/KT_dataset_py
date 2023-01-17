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
import seaborn as sns
import matplotlib as plt
sns.set_style('whitegrid')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
df = pd.concat([train, test])
df.head()
df.info()
sns.countplot(x='Survived', data=df)
sns.countplot(x='Survived', hue='Sex', data=df)
sns.distplot(df['Age'].dropna(), color='Darkred', bins=40)
df2 = df[['Survived','Age']]
df3 = df2.loc[df2['Survived'] == 1]
# df3.head()
df3.info()
sns.distplot(df3['Age'].dropna(), color='Darkred', bins=40)
sns.countplot(x='Embarked', hue='Pclass', data=df)
sns.catplot(x='Embarked', hue='Sex', col='Survived', data=df, kind='count')
sns.countplot(x='Pclass', hue='Survived', data=df)
# sns.countplot(x='Embarked', hue='Survived', data=df)
sns.catplot(x='Embarked', hue='Pclass', col='Survived', data=df, kind='count')
df4 = df[['Survived', 'Embarked', 'Pclass', 'Sex']]
df4 = df4.loc[df4['Survived'] == 1]
df4.head()
sns.catplot(x='Embarked', hue='Pclass', col='Sex', data=df4, kind='count')