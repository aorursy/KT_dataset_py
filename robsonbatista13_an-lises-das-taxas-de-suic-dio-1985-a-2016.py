# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/master.csv')

df.head()
df.columns = ['country', 'year', 'sex', 'age', 'suicides', 'population',

       'suicidesper100kpop', 'country-year', 'HDI for year',

       'gdp_for_year', 'gdp_per_capita', 'generation']

df.columns.values
df.info()
df['gdp_for_year'] = df['gdp_for_year'].str.replace(',','').astype(int)

df.info()
df.shape
df.describe()
df['country'].unique()
#Quantidade de países que estão na base?

df['country'].nunique()
df.isnull().sum()
df = df.drop(['country-year','HDI for year'], axis=1)
df.info()
plt.figure(figsize=(10,10))

sns.lineplot(x='year',y='suicides',data=df)
df_2016 = df[df['year']== 2016]

df_2016
df_2016.info()
df_2016.describe()
print('Os países que estão na lista de 2016 são: %s e a Quantidades de países é %s' %(df_2016['country'].unique(),df_2016['country'].nunique()))
df_2016['age'].unique()
df = df.query('year != 2016')
#As Possíveis faixas de idades

df['age'].unique()
#As possíveis gerações 

df['generation'].unique()
#agrupando os dados e somando por ano

df.groupby(['year']).sum()
#Comparando os suicidios com o PIB per capita e agrupando pelas gerações

plt.figure(figsize=(10,9))

sns.pairplot(x_vars='gdp_per_capita',y_vars='suicides',data=df, height=10,hue='generation')
plt.figure(figsize=(10,9))

sns.barplot(x = 'age', y='suicides', hue='sex', data=df.groupby(["age","sex"]).sum().reset_index()).set_title('Suicidios por faixa etária')

plt.ylabel('SUICÍDIOS')

plt.xlabel('IDADE')

plt.xticks(rotation=90)
plt.figure(figsize=(10,5))

sns.barplot(x = 'age', y='suicidesper100kpop', hue='sex', data=df.groupby(["age","sex"]).sum().reset_index()).set_title('Suicidios por 100.000 pessoas')

plt.ylabel('SUICÍDIOS')

plt.xlabel('IDADE')

plt.xticks(rotation=90)
plt.figure(figsize=(10,9))

sns.barplot(x='sex', y='suicides', hue='generation', data=df)
suic_sum = pd.DataFrame(df['suicides'].groupby(df['country']).sum())

suic_sum = suic_sum.reset_index().sort_index(by='suicides',ascending=False)

most_cont = suic_sum.head(5)

fig = plt.figure(figsize=(10,10))

plt.title('Os 5 países que mais deve casos de suicídios.')

sns.set(font_scale=2)

sns.barplot(y='suicides',x='country',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('Suicídios');

plt.xlabel('Países')

plt.tight_layout()
plt.figure(figsize=(10,10))

plt.title('Gráfico da População por ano')

sns.lineplot(x='year',y='population',hue='sex',data=df)
#Gráfico de correlação das variaveis 

f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(df.corr(), vmin=-1,vmax=1,cmap='coolwarm', annot=True,fmt='.2f', linewidths=.5, ax=ax)