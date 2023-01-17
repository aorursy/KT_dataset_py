import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re
df1=pd.read_csv('../input/google-apps-data-from-kaggle/googleplaystore.csv')

#df2=pd.read_csv('googleplaystore_user_reviews.csv')
df1.head()
df1.info()
df1.Category.unique()
df1.Rating.unique()
df1.Reviews.nunique()
df1.Size.nunique()
df1.Installs.nunique()
df1.Price.unique()
df1['Content Rating'].unique()
df1.Genres.unique()
df1_clean=df1.copy()
df1_clean.drop(df1_clean.index[df1_clean['Category']=='1.9'],inplace=True)
df1_clean.Category.unique()
df1_clean.drop(df1_clean.index[df1_clean['Rating']==19.],inplace=True)
df1_clean.Rating.unique()
df1_clean.dropna(inplace=True)
df1_clean.info()
df1_clean['Installs']=df1_clean['Installs'].str.replace(',','')
df1_clean['Installs']=df1_clean['Installs'].str.replace('+','')
df1_clean['Installs']=df1_clean['Installs'].astype(int)
df1_clean.head()
df1_clean.info()
df1_clean['Reviews']=df1_clean['Reviews'].astype(int)
df1_clean.info()
df1_clean['Price'].unique()
df1_clean['Price']=df1_clean['Price'].str.replace('$','')
df1_clean['Price']=df1_clean['Price'].astype(float)
df1_clean=df1_clean.rename(columns={'Price':'Price(USD)'})
df1_clean.head()
df1_clean.info()
df1_clean['Content Rating'].isnull().sum()
df1_clean.head()
plt.style.use('seaborn')

df1_clean.groupby('Category')['Rating'].mean().sort_values(ascending=False)[:10].plot(kind='barh',color='orange')

plt.title('Top 10 Rated Application Categories',fontweight='bold',fontsize=14)

plt.xlabel('Rate Out of 5',fontsize=12)

plt.ylabel('Apps Categories',fontsize=12)
plt.style.use('seaborn')

csfont = {'fontname':'Comic Sans MS'}

hfont = {'fontname':'Arial'}



df1_clean.groupby('App')['Rating'].mean().sort_values(ascending=False)[:10].plot(kind='barh',color='purple',fontsize=12)

plt.title('Top 10 Rated Applications',**csfont,fontweight='bold',fontsize=14)

plt.xlabel('Rate Out of 5', **hfont)

plt.ylabel('Applications',**hfont)

plt.show()
df1_clean.groupby('App')['Installs'].sum().sort_values(ascending=False)[:10]
plt.style.use('seaborn')

csfont = {'fontname':'Comic Sans MS'}

hfont = {'fontname':'Arial'}



df1_clean.groupby('App')['Installs'].sum().sort_values(ascending=False)[:10].plot(kind='barh',color='green',fontsize=12)

plt.title('Top 10 Installed Applications',**csfont,fontweight='bold',fontsize=14)

plt.xlabel('Number of Installations', **hfont)

plt.ylabel('Applications',**hfont)

plt.show()