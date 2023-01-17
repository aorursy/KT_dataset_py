# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf 
import sklearn

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.columns
df.isnull().sum()#count how mmany missing data points we have 
df.describe().transpose()
sns.pairplot(df)
#plot a distribution of a label 
sns.distplot(df['price'])
#we can deal with prices >2 million-dollars as outliers 
#analysis in categorical columns 
sns.countplot(df['bedrooms'])
#see corelation between columns 
df.corr()
#see correlation with respect to price 
df.corr()['price']
#if we want to sort this 
df.corr()['price'].sort_values()
#we can see that we have some features that are positively correlated or even negatively corr 
#the best thing to see your correlated labels is thrue a scatter plot 
plt.figure(figsize=(10,10))
sns.scatterplot(x='price', y='sqft_living', data=df)
sns.boxplot(x='bedrooms', y='price',data=df)
#the distribution of prices per bedrooms 
#dist of prices per longutude 
sns.scatterplot(x='price',y='long', data=df)
#dist of prices per lat 
sns.scatterplot(x='price',y='lat', data=df)
plt.figure(figsize=(10,10))
sns.scatterplot(x='long', y ='lat',data=df)
#it is very similar to themap of king county of seatel country 
#color theise points darker or whiter with respect to price 
plt.figure(figsize=(10,10))
sns.scatterplot(x='long', y ='lat',data=df, hue='price')
#it is very similar to themap of king county of seatel country 
#clean up this map and delete outliers 
#if wa see our df and discpay most expensive houses
df.sort_values('price', ascending=False).head(20)
#we must just cut off prices >3 miloins dollars: it is just arround 1%
df[df['price']>3000000.0]
len(df[df['price']>3000000.0])
#we will treate them as ouliers 

no_top1_percent=df.sort_values('price',
                               ascending=False).iloc[216:]
no_top1_percent['price'].max()
plt.figure(figsize=(10,10))
sns.scatterplot(x='long', y='lat', data=no_top1_percent,hue='price')
plt.figure(figsize=(10,10))
sns.scatterplot(x='long', y='lat', data=no_top1_percent,
                edgecolor=None,
                alpha=0.2,
                palette='RdYlGn',
                hue='price')
#now we can see where the most expesive hoses are 
#distribution of prices weather the house is in the waterfront or not 
sns.boxplot(x='waterfront',y='price',data=df)
df
#delete unnesesary col
df=df.drop('id',axis=1)
df
df['date']
df['date']=pd.to_datetime(df['date'])
df['date']
df['year']=df['date'].apply(lambda date: date.year)
df['month']=df['date'].apply(lambda date: date.month)
df.head()
sns.boxplot(x='month', y='price',data=df)
df.groupby('month').mean()['price']
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df.drop('date',axis=1)
df.zipcode
df.zipcode.unique()
#how many unique zip codes 
df.zipcode.value_counts()
#their is 0 unique zip codes 
#treate zipcode as categorical 
df=df.drop('zipcode', axis=1)
df
df['yr_renovated'].value_counts()
#0 is not actualy a year --not renovated at all 
df['sqft_basement'].value_counts().sort_values()
