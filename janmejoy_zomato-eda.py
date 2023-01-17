import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline


df = pd.read_csv('../input/zomato.csv',encoding = "ISO-8859-1")
df
df.shape
df.info()
df.isnull().sum()
df1 = df["Cuisines"].describe() 
df1
df1.mode
df["Cuisines"].value_counts()
df["Cuisines"].fillna("unknown", inplace = True)
df.head()
df_numerical   = df.select_dtypes(include = [np.number]).columns

df_categorical = df.select_dtypes(include= [np.object]).columns
df_numerical
df_categorical
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)
plt.figure(figsize=(5,5))

sns.heatmap(df.corr())
df['Is delivering now'].value_counts()
sns.countplot(x='Is delivering now',data=df,palette='viridis',order=['Yes','No'])


df['Has Table booking'].value_counts()
sns.countplot(x='Has Table booking',data=df,palette='viridis',order=['Yes','No'])
df['Has Online delivery'].value_counts()
sns.countplot(x='Has Online delivery',data=df,palette='viridis',order=['Yes','No'])
df['Switch to order menu'].value_counts()

sns.countplot(x='Switch to order menu',data=df,palette='viridis',order=['Yes','No'])
sns.countplot(df['Rating text'])
sns.countplot(df['Price range'])
sns.countplot(df['City'].head(50))
sns.pairplot(df[['Aggregate rating', 'Votes']])
sns.pairplot(df[['Currency', 'Country Code']])
df.hist()
df.plot(kind='density',subplots=True,sharex=False)

plt.show()
df.plot(kind='box',subplots=True,sharex=False,sharey=False)
pd.plotting.scatter_matrix(df)