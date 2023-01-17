import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
cacao = pd.read_csv('../input/flavors_of_cacao.csv')
cacao.info()
cacao.describe()
cacao.head()
type(cacao['Cocoa\nPercent'][0])
cacao['Cocoa\nPercent'] = cacao['Cocoa\nPercent'].apply(lambda x: float(x.split('%')[0]))
type(cacao['Cocoa\nPercent'][0])
cacao.isnull().head()
sns.heatmap(cacao.isnull(),cbar=False,yticklabels=False,cmap='viridis')
cacao['Bean\nType'][0]
len(cacao[cacao['Bean\nType']=='\xa0']) / len(cacao['Bean\nType'])
cacao['Bean\nType'].value_counts().head(10)
plt.figure(figsize=(8,6))
cacao['Broad Bean\nOrigin'].value_counts().head(20).plot.bar()
plt.title('Top 15 Cocoa Bean Producing Nations')
plt.show()
print('Top 15 Cocoa Bean Producing Nations\n')
print(cacao['Broad Bean\nOrigin'].value_counts().head(20))
print('Least common bean location representation\n')
print(cacao['Broad Bean\nOrigin'].value_counts().tail(20))
plt.figure(figsize=(8,6))
cacao['Company\nLocation'].value_counts().head(20).plot.bar()
plt.title('Top 15 Nations by Number of Chocolate Bar Companies')
plt.show()
print('Top 15 Nations by Number of Chocolate Bar Companies\n')
print(cacao['Company\nLocation'].value_counts().head(20))
plt.figure(figsize=(8,6))
cacao['Company\xa0\n(Maker-if known)'].value_counts().head(20).plot.bar()
plt.title('Top Companies by # of Bars Made')
plt.show()
print('Top Companies by # of Bars Made\n')
print(cacao['Company\xa0\n(Maker-if known)'].value_counts().head(20))
plt.figure(figsize=(8,6))
sns.distplot(cacao['Rating'],kde=False,color='green')
plt.title('Ratings Distribution')
plt.show()
print(cacao['Rating'].value_counts())
plt.figure(figsize=(8,6))
sns.distplot(cacao['Review\nDate'],bins=50,kde=False,color='green')
plt.title('Number of Reviews per Year')
plt.show()
sns.pairplot(cacao)
cacao.plot.scatter(x='Cocoa\nPercent',y='Rating')
plt.title('Ratings to % of Cocoa Content')
plt.show()
print('Cocoa % to Quantity ')
print('------------')
print(cacao['Cocoa\nPercent'].value_counts().head(10))
sns.jointplot(x='Cocoa\nPercent',y='Rating',data=cacao,size=8,kind='kde',cmap='coolwarm')
cacao.plot.scatter(x='Review\nDate',y='Rating')
plt.title('Ratings by Review Date')
plt.show()
print('Year and number of ratings that year.\n')
print(cacao['Review\nDate'].value_counts().head(10))
sns.jointplot(x='Review\nDate',y='Rating',data=cacao,size=8,kind='kde',cmap='coolwarm')
cacao[cacao['Rating']==5]
cacao[cacao['Rating']==1]
sns.heatmap(cacao.corr(),cmap='coolwarm',annot=True)
koko = cacao.groupby('Company\nLocation').mean()
plt.figure(figsize=(10,8))
koko['Rating'].plot.bar()
plt.title('Average Rating by Company Location')
plt.tight_layout()
plt.show()
print('Best Average Ratings by Company Location\n')
print(koko['Rating'].sort_values(ascending=False).head(10))
print('Chile has ' + str(len(cacao[cacao['Company\nLocation']=='Chile'])) +' chocolate companies.')
print('The Netherlands have ' + str(len((cacao[cacao['Company\nLocation']=='Amsterdam'] + cacao[cacao['Company\nLocation']=='Netherlands']))) + ' chocolate companies.')
print('The Philippines have ' + str(len(cacao[cacao['Company\nLocation']=='Philippines'])) +' chocolate companies.')
plt.figure(figsize=(10,8))
koko['Cocoa\nPercent'].plot.bar()
plt.title('Average Cocoa % by Company Location')
plt.tight_layout()
plt.show()
print('Highest Average Cocoa % by company Location\n')
print(koko['Cocoa\nPercent'].sort_values(ascending=False).head(10))
print('\n')
print('Lowest Average Cocoa % by company Location\n')
print(koko['Cocoa\nPercent'].sort_values(ascending=False).tail(10))
print('Top Rated by Location of Bean.\n')
print(cacao.groupby('Broad Bean\nOrigin')['Rating'].mean().sort_values(ascending=False).head(10))
print('\n')
print('Bottom Rated by Location of Bean.\n')
print(cacao.groupby('Broad Bean\nOrigin')['Rating'].mean().sort_values(ascending=False).tail(10))
plt.figure(figsize=(10,8))
bean_type = cacao.groupby('Bean\nType')
bean_type.mean()['Rating'].plot.bar()
plt.title('Rating by Bean Type')
plt.tight_layout()
plt.show()
print('Most Common Bean Type\n')
print(cacao['Bean\nType'].value_counts().head(15))
print('\n')
print('Top Rated Beans by Bean Type\n')
print(cacao.groupby(['Bean\nType'])['Rating'].mean().sort_values(ascending=False).head(15))
print('\n')
print('Bottom Rated Beans by Bean Type\n')
print(cacao.groupby(['Bean\nType'])['Rating'].mean().sort_values(ascending=False).tail(15))
