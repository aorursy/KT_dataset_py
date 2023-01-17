import pandas as pd

import numpy as np

import seaborn as sns
df=pd.read_csv('../input/winemag-data-130k-v2.csv')
df.head(5)
df.columns
missing_data=df.isnull()
missing_data.head()
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
df.drop(columns={'designation', 'region_1', 'region_2'}, inplace=True)
df.head()
df.drop(columns={'Unnamed: 0'}, inplace=True)
df.drop(columns={'taster_name', 'taster_twitter_handle'}, inplace=True)
df.head()
avg_price=df['price'].astype('float').mean(axis=0)
df['price'].replace(np.nan, avg_price, inplace=True)
missing_data[['country', 'province']]
df.dropna(subset=['country'], axis=0, inplace=True)
df.dropna(subset=['province'], axis=0, inplace=True)
df.head(5)
df.dtypes
import matplotlib.pyplot as plt

%matplotlib inline
df.corr()
fig, ax = plt.subplots(figsize=(40,30))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Price & Points Linear Correlation', fontweight="bold", size=30)

ax.set_ylabel('Points', fontsize = 30)

ax.set_xlabel('Price', fontsize = 30)

sns.regplot(x='price', y='points', data=df)
sns.boxplot(x='country', y='points', data=df)
df_country=df[['country', 'points', 'price']]
df_country.head()
df_country=df_country.groupby(['country'], as_index=False).mean()

df_country
df_country.dtypes
df_country['points'].idxmax()
df_country['points'].idxmin()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['points'], df['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
df.describe()
df_wine=df[['country', 'variety', 'points', 'price']]
df_wine.head()
df_wine.set_index('country', inplace=True)
df_wine.head()
print(df_wine.loc['US'])
condition=df_wine[['price']]<15.0
df_wine[condition]
df_wine_2=df_wine[(df_wine['variety']=='Cabernet Sauvignon') & (df_wine['price']<15.0) & (df_wine['points']>89)]
df_wine_2.head(100)
df_wine_3=df_wine[(df_wine['variety']=='Pinot Noir') & (df_wine['price']<15.0) & (df_wine['points']>87)]
df_wine_3.head(20)
df_wine_3=df_wine[(df_wine['variety']=='Pinot Noir') & (df_wine['price']<15.0) & (df_wine['points']>90)]

df_wine_3.head(100)
df_wine_4=df_wine[(df_wine['variety']=='Pinot Gris') & (df_wine['price']<15.0) & (df_wine['points']>90)]
df_wine_4.head(100)
df_wine_5=df_wine[(df_wine['variety']=='Chardonnay') & (df_wine['price']<15.0) & (df_wine['points']>90)]
df_wine_5.head(100)
df_wine_6=df_wine[(df_wine['variety']=='Merlot') & (df_wine['price']<15.0) & (df_wine['points']>90)]
df_wine_6.head(100)
df_wine
df_wine_7=df_wine[(df_wine['variety']=='Malbec') & (df_wine['price']<15.0) & (df_wine['points']>90)]
df_wine_7.head(100)
df_wine_8=df_wine[(df_wine['variety']=='Shiraz') & (df_wine['price']<15.0) & (df_wine['points']>90)]
df_wine_8.head(100)
df.head(5)
df_from=df[['country', 'variety', 'points', 'province']]
df_from.head()
df_from.set_index('country', inplace=True)
df_from.head()
american=df_from.loc['US']
american.head()
fig, ax = plt.subplots(figsize=(40,20))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Points', fontweight="bold", size=30)

ax.set_ylabel('Province', fontsize = 30)

ax.set_xlabel('Points', fontsize = 30)

sns.boxplot(x='points', y='province', data=american)
fig, ax = plt.subplots(figsize=(20,100))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=american)
fig, ax = plt.subplots(figsize=(10,75))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=american)
df_from.head()
fig, ax = plt.subplots(figsize=(40,20))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Points', fontweight="bold", size=30)

ax.set_ylabel('Country', fontsize = 30)

ax.set_xlabel('Points', fontsize = 30)

sns.boxplot(x='points', y='country', data=df)
df_Canada=df[(df['country']=='Canada') & (df['price']<15.0) & (df['points']>87)]
df_Canada.head()
df_India=df[(df['country']=='India') & (df['price']<15.0) & (df['points']>89)]

df_India.head(100)
df_England=df[(df['country']=='England') & (df['price']<15.0) & (df['points']>87)]

df_England.head(100)
df_country.head(50)
italian=df_from.loc['Italy']

italian.head()
fig, ax = plt.subplots(figsize=(10,75))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=italian)
french=df_from.loc['France']

french.head()
fig, ax = plt.subplots(figsize=(10,75))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=french)
german=df_from.loc['Germany']

german.head()
fig, ax = plt.subplots(figsize=(10,75))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=german)
austria=df_from.loc['Austria']

austria.head()
fig, ax = plt.subplots(figsize=(10,75))

plt.xticks(fontsize=10) 

plt.yticks(fontsize=10)

ax.set_title('Points', fontweight="bold", size=10)

ax.set_ylabel('Variety', fontsize = 10)

ax.set_xlabel('Points', fontsize = 10)

sns.boxplot(x='points', y='variety', data=austria)
df_US=df[(df['country']=='US') & (df['price']<15.0) & (df['points']>90)]

df_US.head(100)
df_US[['province','variety']]
df_US=df[(df['country']=='US') & (df['price']<15.0) & (df['points']>91)]

df_US[['province','variety', 'description']]
df_FR=df[(df['country']=='France') & (df['price']<15.0) & (df['points']>91)]

df_FR[['province','variety', 'description']]
df_AU=df[(df['country']=='Austria') & (df['price']<15.0) & (df['points']>91)]

df_AU[['province','variety', 'description']]
df_IT=df[(df['country']=='Italy') & (df['price']<15.0) & (df['points']>91)]

df_IT[['province','variety', 'description']]
df_AUS=df[(df['country']=='Australia') & (df['price']<15.0) & (df['points']>91)]

df_AUS[['province','variety', 'description']]
df_IN=df[(df['country']=='India') & (df['price']<15.0) & (df['points']>91)]

df_IN[['province','variety', 'description']]
df_SW=df[(df['country']=='Switzerland') & (df['price']<15.0) & (df['points']>91)]

df_SW[['province','variety', 'description']]
df_IN=df[(df['country']=='India') & (df['price']<15.0) & (df['points']>90)]

df_IN[['province','variety', 'description']]
df_IT=df[(df['country']=='Italy') & (df['price']<15.0) & (df['points']>90)]

df_IT[['province','variety', 'description']]
description=df[['description', 'points']]

description.head(100)
fig, ax = plt.subplots(figsize=(40,20))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Number of wines per points', fontweight="bold", size=30)

ax.set_ylabel('Number of wines', fontsize = 30)

ax.set_xlabel('Points', fontsize = 30)

description.groupby(['points']).count()['description'].plot(ax=ax, kind='bar')
description=description.assign(description_length=description['description'].apply(len))

description.head()
sns.regplot(x='points', y='description_length', data=description)
df_wine.head()