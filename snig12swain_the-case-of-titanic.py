import numpy as np 
import pandas as pd 

import random as rnd
%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
pd.options.display.max_rows = 100
data = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

data.describe()
data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()
sns.set_palette('Dark2') 

fig1, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
sns.countplot(x='Pclass', data=test_df, ax=ax1)
sns.countplot(x='Sex', data=test_df, ax=ax2)
sns.countplot(x='SibSp', data=test_df, ax=ax3)
sns.countplot(x='Parch', data=test_df, ax=ax4);
sns.countplot(x="Pclass", hue="Sex", data=test_df);
sns.factorplot(x='Pclass', hue='Survived', col='Sex', data=data, kind='count');
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(10,8))
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data);
g = sns.PairGrid(data=data,
                 y_vars="Survived",
                 x_vars=["Pclass", "Sex", "SibSp", "Parch"])
g.map(sns.pointplot);
sns.factorplot(x="Embarked", hue="Survived", col="Sex", data=data, kind="count");
sns.factorplot(x="Embarked", y="Survived", col="Sex", data=data, kind="bar");
sns.pointplot(x='Embarked', y='Survived', hue='Sex', data=data);
test_df.isnull().sum()
sum(pd.isnull(test_df['Age']))
data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()
ax = test_df["Age"].hist(bins=15, color='teal', alpha=0.5)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
test_df["Age"].median(skipna=True)
data[['Age', 'Fare']].describe()
sns.distplot(data['Age'].dropna());
sns.distplot(data['Fare'].dropna());
data['logFare'] = data['Fare'].apply(lambda x: np.log(x + 1))
sns.distplot(data['logFare']);
g = sns.FacetGrid(data, hue='Survived', size=6)
g.map(plt.scatter, 'Age', 'logFare',s=30, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend();
g = sns.lmplot(x='Age', y='logFare', hue='Pclass',
               truncate=True, size=6, data=data[data.Fare != 0])
g = sns.PairGrid(data,
                 x_vars=['Survived'],
                 y_vars=['Age', 'logFare'],
                 size=5)
g.map(sns.violinplot, palette='Dark2');
data['Cabin'].unique()
sns.set_palette('Dark2')
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(6,6))
plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True,
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')

plt.legend()
print(data['Fare'].mean())

figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['b','r'],
         bins = 30 ,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='blue',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',color = 'blue',figsize=(7,8) ,ax = ax)