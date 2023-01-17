import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('seaborn')
df = pd.read_csv('/kaggle/input/food-preferences/Food_Preference.csv')
df.head()
df.info()
df.drop(['Timestamp','Participant_ID'],axis=1,inplace=True)
df.head()
df.isnull().any()
df.isnull().sum()
df=df.dropna()
sns.heatmap(df.isnull())
sns.countplot(x='Dessert', data = df, color = '#C70039')
df['Dessert'] = df['Dessert'].replace('Yes',1)

df['Dessert'] = df['Dessert'].replace('No',0)

df['Dessert'] = df['Dessert'].replace('Maybe',1)
pd.to_numeric(df['Dessert'],errors = 'coerce')
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Dessert',data=df,color='#3ff08a',ax=ax[0])

sns.countplot(x='Food', data = df, color = '#e14735',ax=ax[1])
fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Gender',data=df,color='#4d5ae8',ax=ax[0])

sns.countplot(x='Dessert', data = df, color = '#C70039',ax=ax[1])
fig,ax = plt.subplots(figsize=(10,10))

sns.scatterplot(x='Age',y='Dessert',hue = 'Nationality',ax = ax,color = '#e14735',data=df)

plt.show()
fig,ax = plt.subplots(figsize=(15,7))

sns.countplot(y='Nationality',data = df)

plt.show()
fig,ax = plt.subplots(figsize=(15,17))

sns.countplot(y='Age',data = df)