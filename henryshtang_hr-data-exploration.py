import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/core_dataset.csv')

df.head(50) #first seven data points
df.describe()
df.shape #(rows,columns)
df.columns

df['Sex']
df['Sex'].unique()
df['Sex'].replace('male','Male',inplace=True)
df['Sex'].unique()
df.dropna(subset=['Sex'],inplace=True) # remove rows with nan values for Sex

df.shape

df['Sex'].value_counts() #Lets plot this

df['Sex'].value_counts().plot(kind='bar')
import matplotlib.pyplot as plt



df['Sex'].value_counts().plot(kind='bar')
# Gender diversity across departmets

import seaborn as sns

plt.figure(figsize=(16,9))

ax=sns.countplot(x=df['Department'],hue=df['Sex'])
df['RaceDesc'].value_counts().plot(kind='pie')
df['CitizenDesc'].unique()
df['CitizenDesc'].value_counts().plot(kind='bar')
df['Age'].hist()
df['Age'].hist(bins=40)
df['Position'].unique()
plt.figure(figsize=(16,9))

df['Position'].value_counts().plot(kind='bar')
df['Pay Rate'].describe()
df['Age'].describe()
df.plot(x='Age',y='Pay Rate',kind='scatter')

# Looks like thery are not related! 
df['Manager Name'].unique()
df['Performance Score']
plt.figure(figsize=(20,20))

sns.countplot(y=df['Manager Name'], hue=df['Performance Score'])
df.groupby('Department')['Pay Rate'].sum().plot(kind='bar')

#Production department pays more!
plt.figure(figsize=(16,9))

df.groupby('Position')['Pay Rate'].sum().plot(kind='bar')
id_of_person_with_highgest_pay = df['Pay Rate'].idxmax()

df.loc[id_of_person_with_highgest_pay]





df.loc[df['Pay Rate'].idxmax()]



HispLat_map ={'No': 0, 'Yes': 1, 'no': 0, 'yes': 1}

df['Hispanic/Latino'] = df['Hispanic/Latino'].replace(HispLat_map)

df['Hispanic/Latino']
sns.violinplot('Hispanic/Latino', 'Pay Rate', data = df)
df.columns
df['MaritalDesc'].value_counts().plot(kind='bar')
df['MaritalDesc'].value_counts().plot(kind='bar')
plt.figure(figsize=(16,9))

ax=sns.countplot(x=df['Employment Status'],hue=df['MaritalDesc'])
df2 = pd.read_csv('../input/HRDataset_v9.csv')

df2.head(50) #first seven data points
df2.columns
df2.plot(x='Days Employed',y='Pay Rate',kind='scatter')

sns.violinplot('CitizenDesc', 'Pay Rate', data = df2)
plt.figure(figsize=(16,9))

ax=sns.countplot(x=df2['CitizenDesc'],hue=df2['RaceDesc'])