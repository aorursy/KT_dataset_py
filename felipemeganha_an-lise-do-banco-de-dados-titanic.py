import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import os

print(os.listdir("../input"))



df = pd.read_csv('../input/train_and_test2.csv')
df.head(3)
df.columns
df.drop(['Passengerid','zero', 'zero.1',

       'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6','zero.7',

       'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',

       'zero.14','zero.15', 'zero.16', 'zero.17',

       'zero.18'],axis=1,inplace=True)
df.columns=['age','fare','sex','sibsb','parch','pclass','embarked','survived']
df.head(3)
df.shape
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.fillna(df.mean(), inplace=True)
df.isnull().sum()
sns.pairplot(df)
df.columns
sns.distplot(df['age'], bins=10)
df['age'].mean()
df['survived'].value_counts()
sns.set_style('whitegrid')

sns.countplot(x='survived',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='survived',hue='sex',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='survived',hue='pclass',data=df,palette='rainbow')