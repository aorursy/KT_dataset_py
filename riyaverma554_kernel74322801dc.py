# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import missingno as msno
import squarify

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size':12})
dataset=pd.read_csv('/kaggle/input/wine-reviews/winemag-data_first150k.csv',index_col=0)
dataset1=pd.read_csv('/kaggle/input/wine-reviews/winemag-data-130k-v2.csv',index_col=0)
wine=pd.concat([dataset,dataset1],axis=0)
print('the number of rows are:',wine.shape)
wine.info()
wine.head()
wine.describe(include='all').T
msno.bar(wine,color=sns.color_palette('viridis'))
print('number of country list in data',wine['country'].nunique())
plt.figure(figsize=(14,12))
cnt=wine['country'].value_counts().to_frame()[0:20]
sns.barplot(x=cnt['country'], y=cnt.index, data=cnt, palette='ocean',orient='h')
plt.title('distribution of wine reviews by top 20 countries')

f,ax=plt.subplots(1,2,figsize=(14,6))
ax1,ax2=ax.flatten()

sns.distplot(wine['price'].fillna(wine['price'].mean()), color='r',ax=ax1)
ax1.set_title('distribution of price')

sns.boxplot(x=wine['price'],ax=ax2)
ax2.set_ylabel('')
ax2.set_title('boxplot of price')
cnt=wine.groupby(['country'])['price'].mean().sort_values(ascending=False).to_frame()

plt.figure(figsize=(16,8))
sns.pointplot(x=cnt['price'], y=cnt.index, color='r',orient='h',markers='o')
plt.title('country wise average wine price')
plt.xlabel('price')
plt.ylabel('country')

cnt=wine.groupby(['country'])['price'].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(12,8))

squarify.plot(cnt['price'].fillna(1),color=sns.color_palette('rainbow'),label=cnt.index)
fig,ax=plt.subplots(1,2,figsize=(16,8))
ax1,ax2=ax.flatten()

cnt=wine.groupby(['country'])['price'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x=cnt['price'], y= cnt.index, data=cnt, palette='inferno',ax=ax1)
ax1.set_title('country wise most expensive wine')
ax1.set_ylabel('variety')
ax1.set_xlabel('')

cnt= wine.groupby(['country'])['price'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x=cnt['price'], y=cnt.index, data=cnt, palette='rainbow_r', ax=ax2)
ax2.set_title('country wise least expensive wines')
ax2.set_ylabel('')
ax2.set_xlabel('')
plt.subplots_adjust(wspace=0.3);

plt.figure(figsize=(12,6))
sns.boxplot(x=wine['country'], y=wine['price'])
plt.yscale('log')
plt.title('country wise box plot(log scale)')
plt.xticks(rotation=90)
cnt=wine.groupby(['country'])['points'].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(16,8))

sns.pointplot(x=cnt['points'], y=cnt.index, data=cnt, color='r', orient='h')
plt.title('countrywise')
plt.xlabel('points')
plt.ylabel('countries')

fig,ax= plt.subplots(1,2,figsize=(16,8))
ax1,ax2=ax.flatten()

cnt= wine.groupby(['country'])['points'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x=cnt['points'], y=cnt.index, data=cnt, palette='hot', ax=ax1)
ax1.set_title('highest rated wines')
ax1.set_ylabel('variety')
ax1.set_xlabel('')

cnt=wine.groupby(['country'])['points'].min().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x=cnt['points'], y=cnt.index, data=cnt, palette='ocean', ax=ax2)
ax2.set_title('least rated wines')
ax2.set_ylabel('')
ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.3)
plt.figure(figsize=(16,6))
sns.boxplot(x=wine['country'], y=wine['points'])
plt.title('countrywise boxplot')
plt.xticks(rotation=90)

sns.jointplot(x=wine['points'], y=wine['price'], color='g')
print('number of variety of wines',wine['variety'].nunique())
fig,ax=plt.subplots(1,2,figsize=(16,8))
ax1,ax2=ax.flatten()

cnt=wine.groupby(['variety'])['price'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x=cnt['price'], y=cnt.index, palette='cool', ax=ax1)
ax1.set_title('top most variety of wine(basis of price)')
ax1.set_ylabel('variety')
ax1.set_xlabel('')

cnt=wine.groupby(['variety'])['points'].max().sort_values(ascending=False).to_frame()[:15]
sns.barplot(x=cnt['points'], y=cnt.index, palette='Wistia', ax=ax2)
ax2.set_title('top most variety of wine(basis of ratings)')
ax2.set_ylabel('variety')
ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.5)
print('number of variety of wines',wine['variety'].nunique())
fig,ax=plt.subplots(1,2,figsize=(16,8))
ax1,ax2=ax.flatten()

cnt=wine.groupby(['variety'])['price'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x=cnt['price'], y=cnt.index, palette='ocean_r', ax=ax1)
ax1.set_title('least variety of wines(basis of price)')
ax1.set_ylabel('variety')
ax1.set_xlabel('')

cnt=wine.groupby(['variety'])['points'].min().sort_values(ascending=True).to_frame()[:15]
sns.barplot(x=cnt['points'], y=cnt.index, palette='rainbow', ax=ax2)
ax2.set_title('least variety of wines(basis of ratings)')
ax2.set_ylabel('variety')
ax2.set_xlabel('')

plt.subplots_adjust(wspace=0.5)
cnt=wine.groupby(['country','points'])['price'].agg(['count','min','max','mean']).sort_values(by='mean',ascending=False)[:10]
cnt.reset_index(inplace=True)
cnt.style.background_gradient(cmap='magma',high=0.5)

