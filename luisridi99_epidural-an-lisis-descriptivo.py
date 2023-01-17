!pip install pyreadstat
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
df = pd.read_spss('/kaggle/input/epidural.sav')

df.head()
df[{"edad","PESONAC"}].describe()
df[{"TEMP1","TEMP2"}].describe()
df["TIPOPAR"].value_counts()
df["TIPOPAR"].value_counts().apply(lambda x: (x/df["TIPOPAR"].value_counts().sum())*100)
plt.figure(figsize=(5,5))

plt.title('Analisis tipo de parto')

locs, labels = plt.xticks()

sns.countplot(df["TIPOPAR"])
df["OXITOCIN"].value_counts()
df["OXITOCIN"].value_counts().apply(lambda x: (x/df["OXITOCIN"].value_counts().sum())*100)
plt.figure(figsize=(5,5))

plt.title('Analisis uso de Oxitocina')

locs, labels = plt.xticks()

sns.countplot(df["OXITOCIN"])
pd.value_counts(df['EPIDURAL'])
100 * df['EPIDURAL'].value_counts() / len(df['EPIDURAL'])
plot = df['EPIDURAL'].value_counts().plot(kind='pie', autopct='%.2f', 

                                            figsize=(6, 6),

                                            title='Anestesia en mujeres')
df[{"PESONAC","EPIDURAL"}].groupby(['EPIDURAL']).describe()
df[{"edad","EPIDURAL"}].groupby(['EPIDURAL']).describe()
df[{"TEMP1","EPIDURAL"}].groupby(['EPIDURAL']).describe()
df[{"TEMP2","EPIDURAL"}].groupby(['EPIDURAL']).describe()
pd.crosstab(index=df['EPIDURAL'],columns=df['TIPOPAR'], margins=True)
pd.crosstab(index=df['EPIDURAL'],columns=df['TIPOPAR']).apply(lambda r: r/r.sum() *100,axis=0)
plot = pd.crosstab(index=df['EPIDURAL'],

            columns=df['TIPOPAR']).plot(kind='bar')
pd.crosstab(index=df['EPIDURAL'],columns=df['OXITOCIN'], margins=True)
pd.crosstab(index=df['EPIDURAL'],columns=df['OXITOCIN']).apply(lambda r: r/r.sum()*100,axis=0)
plot = pd.crosstab(index=df['EPIDURAL'],columns=df['OXITOCIN']).plot(kind='bar')
df_dummies = pd.get_dummies(df['OXITOCIN'])

#del df_dummies[df_dummies.columns[-1]]

df_dummies.head()
df_new = pd.concat([df, df_dummies], axis=1)

del df_new['OXITOCIN']

df_new.head()
m=df_new.loc[:].corr()

plt.figure(figsize=(10,10))

sns.heatmap(m,annot=True,cmap="Reds")