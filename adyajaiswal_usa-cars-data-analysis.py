import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv',index_col=0)
data.head(10)    
data.info()
data.describe()
median=data.groupby('brand')['price'].median()

def fill_median(cols):

    price=cols[0]

    brand=cols[1]

    if price==0:

        return median[brand]

    else:

        return price
data['price']=data[['price','brand']].apply(fill_median,axis=1)
sns.set_style('whitegrid')

plt.figure(figsize=(8,5))

sns.distplot(data['price'])
data[data['price']>65000]
brands= data.groupby('brand')['model'].count().sort_values(ascending = False).head(10)

brands=brands.reset_index()

fig=plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(brands['brand'],brands['model'],color='teal')

ax.set_xlabel('Brand')

ax.set_ylabel('Count')

colors=data.groupby('color')['brand'].count().sort_values(ascending=False).head(10)

colors=colors.reset_index()

fig=plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(colors['color'],colors['brand'],color='pink')

ax.set_xlabel('Color')

ax.set_ylabel('Count')
years= data.groupby('year')['model'].count().sort_values(ascending = False).head(10)

years=years.reset_index()

fig=plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(years['year'],brands['model'],color='purple')

ax.set_xlabel('Year')

ax.set_ylabel('Count')
plt.figure(figsize=(8,6))

sns.heatmap(data.corr(),annot=True)
plt.figure(figsize=(8,5))

sns.scatterplot(x='year',y='price',data=data,hue='title_status',alpha=0.2)
plt.figure(figsize=(8,6))

sns.scatterplot(x='mileage',y='price',data=data,color='purple')
sns.pairplot(data[['price','year','mileage']])

plt.tight_layout()