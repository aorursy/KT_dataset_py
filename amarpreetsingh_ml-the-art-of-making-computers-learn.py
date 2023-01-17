import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/train.csv',encoding = "ISO-8859-1",low_memory=False)
df.head()
df.columns.values
# Data Analysis and Exploration

sns.countplot('Sex',hue='Survived',data=df)

plt.show()
# Age as a Categorical feature

df['Age_band']=0

df.loc[df['Age']<=16,'Age_band']=0

df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1

df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2

df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3

df.loc[df['Age']>64,'Age_band']=4

df.head(2)
df['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')#checking the number of passenegers in each band
# Cleaning data

df.Age.isnull().sum()

#df.loc[
# Converting String Values into Numeric

df['Sex'].replace(['male','female'],[0,1],inplace=True)
pd.crosstab(df.Pclass,df.Survived,margins=True).style.background_gradient(cmap='summer_r')
df['Title'] = None

for index,row in enumerate(df['Name']):

    title = row.split(', ')[1].split('. ')[0]

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Mr','Rev', 'Sir']:

        df.loc[index, 'Title'] = 'Mr'

    elif title in [ 'Ms', 'Mme', 'Mrs', 'the Countess','Lady']:

        df.loc[index, 'Title'] = 'Mrs'

    elif title in ['Master']:

        df.loc[index, 'Title'] = 'Master'

    elif title in ['Miss','Mlle']:

        df.loc[index, 'Title'] = 'Ms'

    else:

        df.loc[index, 'Title'] = 'Other'
pd.crosstab(df.Title,df.Survived,margins=True).style.background_gradient(cmap='summer_r')
df.groupby(['Sex','Survived'])[['Survived']].count().plot(kind='bar')
df
sns.countplot('Sex', hue='Survived',data=df)
df['Sex'].replace(['female','male'],[1,0],inplace=True)
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
df['Survived'].mean()