# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/world-happiness/2015.csv')
df.head()
df.info()
df.describe().T
df.sort_values('Happiness Rank').head(10)
df.sort_values('Happiness Score',ascending=False).head(10)
df.sort_values('Dystopia Residual',ascending=False).head(10)
df['Region'].unique()
plt.figure(figsize=(9,6))

sns.countplot(df['Region'])

plt.xticks(rotation=90);
sns.pairplot(df.drop('Happiness Rank',axis=1));
fig,ax=plt.subplots(3,1,figsize=(6,12))

_=sns.regplot(df['Economy (GDP per Capita)'],df['Happiness Score'],ax=ax[0])

_=sns.regplot(df['Family'],df['Happiness Score'],ax=ax[1])

_=sns.regplot(df['Health (Life Expectancy)'],df['Happiness Score'],ax=ax[2])

plt.show()
plt.figure(figsize=(9,6))

sns.regplot(df['Economy (GDP per Capita)'],df['Health (Life Expectancy)'])

plt.show()
plt.figure(figsize=(9,6))

sns.barplot(df['Country'][:10],df['Economy (GDP per Capita)'])

plt.xticks(rotation=90);
plt.figure(figsize=(9,6))

sns.barplot(df['Country'][:10],df['Health (Life Expectancy)'])

plt.xticks(rotation=90);
plt.figure(figsize=(9,6))

sns.barplot(df['Country'][:10],df['Freedom'])

plt.xticks(rotation=90);
plt.figure(figsize=(9,6))

sns.barplot(df['Region'],df['Trust (Government Corruption)'])

plt.xticks(rotation=90);
generosity=df.sort_values(by="Generosity",ascending="True")[:20].reset_index()

generosity=generosity.drop('index',axis=1)



plt.figure(figsize=(11,6))

sns.barplot(x=generosity.Region.value_counts().index,y=generosity.Region.value_counts().values)

plt.xlabel("Region")

plt.ylabel("Count")

plt.xticks(rotation=90)

plt.title("Top most generous region rates")

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(df['Economy (GDP per Capita)'], df['Trust (Government Corruption)'], s=(df['Happiness Score']**3))

plt.xlabel("Economy")

plt.ylabel("Trust");
plt.figure(figsize=(10,10))

plt.scatter(df['Economy (GDP per Capita)'], df['Freedom'], s=(df['Happiness Score']**3))

plt.xlabel("Economy")

plt.ylabel("Freedom");
plt.figure(figsize=(10,10))

plt.scatter(df['Freedom'], df['Health (Life Expectancy)'], s=(df['Happiness Score']**3))

plt.ylabel("Health")

plt.xlabel("Freedom");
plt.figure(figsize=(12,6))

sns.boxplot(df['Region'],df['Dystopia Residual'])

plt.xticks(rotation=90);