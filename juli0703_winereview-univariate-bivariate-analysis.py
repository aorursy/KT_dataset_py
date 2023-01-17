import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
wine = pd.read_json('../input/winemag-data-130k-v2.json')
wine.shape
wine.describe()
wine.head()
wine.isnull().head()
sns.heatmap(wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')
wine['price'].isnull().value_counts()
wine = wine.drop(['region_1','region_2','taster_twitter_handle','designation','taster_name'],axis=1)
sns.heatmap(wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')
wine.head()
wine = wine.dropna()
wine.shape
wine.nunique()
wine[wine.duplicated('title',keep=False)].sort_values('title').head(10)
wine[wine.duplicated('description',keep=False)].sort_values('description').head(10)
len(wine[wine.duplicated('description',keep=False)].sort_values('description'))
wine = wine.drop_duplicates('description')
wine.shape
wine.nunique()
wine['desc_length'] = wine['description'].apply(len)
plt.figure(figsize=(10,6))
wine['country'].value_counts().head(10).plot.bar()
plt.title('Top Ten Wine Producing Countries by Varieties Produced.')
plt.show()
print('Top Ten Wine Producing Countries by Varieties Produced.\n')
print(wine['country'].value_counts().head(10))
plt.figure(figsize=(10,6))
sns.distplot(wine['points'],kde=False,color='orange')
plt.title('Frequency of Points Given')
plt.show()
print('Frequency of Points Given\n')
print(wine.groupby('points')['points'].count())
plt.figure(figsize=(10,6))
#sns.distplot()
sns.distplot(wine[wine['price']<=200]['price'],kde=False,bins=50,color='purple')
plt.title('Price Distribution for Wines under $200')
plt.show()
#wine[wine['price']<=200]['price']
plt.figure(figsize=(10,6))
wine['province'].value_counts().head(10).plot.bar()
plt.title('Top Ten Wine Producing Provinces by Variety Produced.')
plt.show()
print('Top Ten Wine Producing Provinces by Variety Produced.\n')
print(wine['province'].value_counts().head(10))
print('Ratio of U.S. wine variety from California: ' + str(len(wine[wine['province']=='California']) / len(wine[wine['country']=='US'])))
print('Ratio of World wine variety from California: ' + str(len(wine[wine['province']=='California']) / len(wine)))
plt.figure(figsize=(10,6))
wine['variety'].value_counts().head(10).plot.bar()
plt.title('Top Ten Wines by Variety')
plt.show()
print('Top Ten Wines by Variety\n')
print(wine['variety'].value_counts().head(10))
plt.figure(figsize=(10,6))
sns.distplot(wine['desc_length'],bins=100,kde=False)
plt.title('Length of Characters per Taster Description')
plt.xlabel('Characters per Desc.')
plt.ylabel('# of Desc.')
plt.show()
sns.heatmap(wine.corr(),annot=True,cmap='plasma')
plt.figure(figsize=(10,6))
sns.pairplot(wine)
plt.show()
sns.jointplot(x='points',y='price',data=wine)
wine[wine['price']>=3000]
sns.jointplot(x='points',y='price',data=wine, kind='kde', cmap='plasma')
sns.lmplot(x='points',y='price',data=wine[wine['price']<=200])
sns.jointplot(x='points',y='price',data=wine[wine['price']<=200], kind='kde',cmap='plasma')
sns.jointplot(x='points',y='desc_length',data=wine)
print(wine['description'][97446])
plt.figure(figsize=(10,6))
sns.boxplot(x='points', y='desc_length', data=wine)
plt.title('Points given by Length of Review')
plt.show()
wine[wine['price']>=1000]['variety'].value_counts().plot.bar()
plt.title('Varieties of Wine over $1000')
plt.show()
print('Varieties of Wine over $1000.\n')
print(wine[wine['price']>=1000]['variety'].value_counts())
wine[wine['price']>=1000]['country'].value_counts().plot.bar()
plt.title('Number of Wines over $1000 by Country')
plt.show()
print('Number of Wines over $1000 by Country.\n')
print(wine[wine['price']>=1000]['country'].value_counts())
plt.figure(figsize=(8,6))
wine[wine['price']<=5]['variety'].value_counts().head(10).plot.bar()
plt.title('Varieties of Wine under $6')
plt.show()
print('Varieties of Wine under $6.\n')
print(wine[wine['price']<=5]['variety'].value_counts().head(10))
plt.figure(figsize=(8,6))
wine[wine['price']<=5]['country'].value_counts().head(10).plot.bar()
plt.title('Top Countries with Wines under $6')
plt.show()
print('Top Countries with Wines under $6.\n')
print(wine[wine['price']<=5]['country'].value_counts().head(10))
plt.figure(figsize=(8,6))
wine[wine['points']>=98]['variety'].value_counts().head(10).plot.bar()
plt.title('Varieties of Wines that scored 98 points or more.')
plt.show()
print('Varieties of Wines that scored 98 points or more.\n')
print(wine[wine['points']>=98]['variety'].value_counts().head(10))
plt.figure(figsize=(8,6))
wine[wine['points']<=82]['variety'].value_counts().head(10).plot.bar()
plt.title('Varieties of Wines that scored 82 points or less')
plt.show()
print('Varieties of Wines that scored 82 points or less.\n')
print(wine[wine['points']<=82]['variety'].value_counts().head(10))
wine_country = wine.groupby('country')
plt.figure(figsize=(14,6))
sns.boxplot(x='country',y='points',data=wine)
plt.title('Average Points Given by Country')
plt.xticks(rotation = 90)
plt.show()
print('Average points given by Country\n')
print(wine_country['points'].mean().sort_values(ascending=False).head(15))
big_wine = wine.groupby('country').filter(lambda x: len(x) > 100)
plt.figure(figsize=(14,6))
sns.boxplot(x='country',y='points',data=big_wine)
plt.title('Average Points Given by Country')
plt.xticks(rotation = 90)
plt.show()
print('Average points given by Country:\n')
#print(big_wine['points'].mean().sort_values(ascending=False))
print(big_wine.groupby('country')['points'].mean().sort_values(ascending=False).head(15))
wine.describe()
economyWine = wine[(wine['points'] >= 91) & (wine['price'] <= 17)] 
plt.figure(figsize=(8,6))
economyWine['country'].value_counts().head(10).plot.bar()
plt.title('Economy Wines by Country')
plt.show()
print('There are ' + str(len(economyWine)) + ' economy wines.')
print('Economy Wines by Country:\n')
print(economyWine['country'].value_counts().head(10))
plt.figure(figsize=(8,6))
economyWine['variety'].value_counts().head(10).plot.bar()
plt.title('Economy Wines by Variety')
plt.show()
print('Economy Wines by Variety:\n')
print(economyWine['variety'].value_counts().head(10))
economyWine.describe()
superEcon = economyWine[(economyWine['points'] >= 92) & (economyWine['price'] <= 14)]
plt.figure(figsize=(8,6))
superEcon['country'].value_counts().plot.bar()
plt.title('Super Economy Wines by Country')
plt.show()
print('There are ' + str(len(superEcon)) + ' super economy wines.')
print('Super Economy Wines by Country:\n')
print(superEcon['country'].value_counts())
plt.figure(figsize=(8,6))
superEcon['variety'].value_counts().plot.bar()
plt.title('Mega Value Wines by Variety')
plt.show()
print('Mega Value wines by Variety\n')
print(superEcon['variety'].value_counts())