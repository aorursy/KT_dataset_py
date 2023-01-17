import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
titanicSet = pd.read_csv('../input/Titanictrain.csv')

titanicSet.head()
titanicSet.info()
titanicSet.corr()
titanicSet.describe()
titanicSet.describe(include=np.object)
titanicSet['Age'] = titanicSet.groupby(['Pclass','Embarked','Sex'])['Age'].apply(lambda x: x.fillna(x.mean()))
titanicSet[titanicSet['Age'].isnull()]
titanicSet['Age'] = titanicSet.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.mean()))
titanicSet[titanicSet['Age'].isnull()]
titanicSet.info()
titanicSet['Embarked'].mode()

titanicSet['Embarked'].fillna('S',inplace=True)
titanicSet.info()
titanicSet['Cabin'].fillna(method='ffill',inplace=True)
titanicSet['Cabin'].isnull().sum()
titanicSet['Cabin'].fillna(method='bfill',inplace=True)
titanicSet['Cabin'].isnull().sum()
titanicSet.info()
sns.countplot(titanicSet['Survived'])

plt.show()
sns.distplot(titanicSet['Fare'],kde=False)

plt.show()
sns.distplot(titanicSet['Age'])

plt.show()
sns.countplot(titanicSet['Sex'])

plt.show()
sns.countplot(titanicSet['Pclass'])

plt.show()
sns.countplot(titanicSet['Sex'],hue=titanicSet['Survived'])

plt.show()
sns.countplot(titanicSet['Pclass'],hue=titanicSet['Survived'])

plt.show()
sns.countplot(titanicSet['Embarked'],hue=titanicSet['Survived'])

plt.show()
'''S less likely to survive'''
plt.figure(figsize=(15,5))

sns.boxplot(titanicSet['Survived'],titanicSet['Fare'])

plt.show()
'''survived people have paid more fare.'''
plt.figure(figsize=(15,5))

sns.boxplot(titanicSet['Survived'],titanicSet['Age'])

plt.show()
'''age doesnt have much effect on survival'''
plt.figure(figsize=(15,5))

sns.countplot(titanicSet['Parch'],hue=titanicSet['Survived'])

plt.show()
'''0 companion have less survival chance'''
plt.figure(figsize=(15,5))

sns.scatterplot(x=titanicSet['Age'],y=titanicSet['Fare'],hue=titanicSet['Survived'])

plt.show()
'''survived people have paid more or age is less compared to not survived'''
plt.figure(figsize=(15,5))

sns.swarmplot(x=titanicSet['Sex'],y=titanicSet['Fare'],hue=titanicSet['Survived'])

plt.show()
'''Survival rate of female or higher fare is more.'''
plt.figure(figsize=(15,5))

sns.heatmap(titanicSet.corr(),annot=True)

plt.show()