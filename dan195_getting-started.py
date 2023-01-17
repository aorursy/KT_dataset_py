

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/database.csv')

df.head()
df.shape
df.body_camera.value_counts()
df.manner_of_death.value_counts()
df.armed.value_counts()
df.gender.value_counts()
df.race.value_counts()
sns.countplot(df['race'],hue=df['threat_level'],palette='Blues_d')
df.flee.value_counts()
df.isnull().sum()
DF = df.dropna()

DF.shape
sns.distplot(DF['age'],kde=True,rug=True)
DF['age'].describe()
minor_by_gen = df.loc[df['age'] < 18].gender.value_counts()

minor_by_eth = df.loc[df['age'] < 18].race.value_counts()

n_minors = len(df.loc[df['age'] < 18])

print(minor_by_gen, '\n\n{}'.format(minor_by_eth), '\n\nThere were {} minors who were shot.' .format(str(n_minors)))
df.loc[df['age'] < 18].armed.value_counts()
sns.countplot(df.loc[df['age'] < 18]['armed'],palette = 'Paired')
unarmed_youth_df = df.loc[(df['age'] < 18) & ((df['armed'] == 'unarmed') | (df['armed'] == 'toy weapon'))]

unarmed_youth_df.shape
sns.countplot(unarmed_youth_df['race'],hue=unarmed_youth_df['threat_level'],palette='bright')
sns.countplot(unarmed_youth_df.loc[unarmed_youth_df['armed']=='unarmed']['race'],palette='bright')
df.loc[(df['age'] < 31) & (df['age'] > 17)].race.value_counts()
df.loc[(df['age'] < 31) & (df['age'] > 17)].armed.value_counts()
unarmed_ya_df = df.loc[((df['age'] < 31) & (df['age'] > 17)) & ((df['armed']=='unarmed') | (df['armed']=='toy weapon'))]

unarmed_ya_df.shape
unarmed_ya_df.flee.value_counts()
sns.countplot(unarmed_ya_df['race'],palette='bright')
sns.countplot(unarmed_ya_df[unarmed_ya_df['armed']=='unarmed']['race'],palette='bright')
sns.countplot(unarmed_ya_df.loc[(unarmed_ya_df['flee']=='Not fleeing') & (unarmed_ya_df['armed']=='unarmed')]['race'],palette='bright')
df.loc[df.body_camera==True].race.value_counts()
df.loc[df.body_camera==True].armed.value_counts()
sns.countplot(y='state',data=df,palette='Blues_d')