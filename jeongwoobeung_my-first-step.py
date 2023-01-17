import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=2.5)



import missingno as msno

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.describe()
df_train.describe()
df_test.describe()
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum()/df_train[col].shape[0]))

    print(msg)
msno.matrix(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2)) #iloc = index location
f, ax = plt.subplots(1,2, figsize=(18,8))



df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct = '%1.1f%%', ax = ax[0], shadow = True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')

plt.show()
df_train.shape
df_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=True).count()
df_train[['Pclass','Survived']].groupby(['Pclass']).count()
df_train['Survived'].unique()
pd.crosstab(df_train['Pclass'], df_train['Survived'],margins=True).style.background_gradient(cmap='summer_r')
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot()
y_position = 1.02

r, ax = plt.subplots(1,2, figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00','#D3D3D3'],ax = ax[0])

ax[0].set_title('Number of passengers By Pclass',y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived', data=df_train,ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead',y=  y_position)

plt.show()
f, ax = plt.subplots(1,2, figsize=(18,8))

df_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue = 'Survived', data=df_train, ax= ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()

pd.crosstab(df_train['Sex'], df_train['Survived'],margins=True).style.background_gradient('summer')
sns.factorplot('Pclass','Survived',hue='Sex', data=df_train, size=6, aspect=2.0)
sns.factorplot(x='Sex', y= 'Survived', hue='Pclass',data = df_train, saturation=.5, size=9, aspect =1)