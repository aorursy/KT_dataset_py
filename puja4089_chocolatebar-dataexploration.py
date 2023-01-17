import pandas as pd

import numpy as np

%matplotlib notebook

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv(r"../input/flavors_of_cacao.csv",na_values = '\xa0')
data.head()
data.isnull().sum()
plt.figure(figsize = (8,6))

sns.distplot(data['Rating'],bins = 5, color = 'green')
data.describe()
data['coco % as num'] = data['Cocoa\nPercent'].apply(lambda x: x.split('%')[0])

data['coco % as num'] = data['coco % as num'].apply(lambda z : float(z))
data.head()
plt.figure(figsize = (8,6))

sns.distplot(data['coco % as num'],bins = 20, color = 'brown')
type(data['Review\nDate'][0])
years = set(data['Review\nDate'])
years
year_counts = data['Review\nDate'].value_counts()
year_counts
data['Review\nDate'] = data['Review\nDate'].astype(str)



plt.figure(figsize=(13,8))

sns.boxplot(x='Review\nDate', y='Rating',data=data)
bean_origin = set(data['Broad Bean\nOrigin'] )
a = data.groupby(['Broad Bean\nOrigin'])['Rating'].mean()

a = a.sort_values(ascending = False)
df = pd.DataFrame({'Broad Bean\nOrigin':a.index,'Rating':a.values})
df.head()
plt.figure(figsize = (15,20))

ax = sns.barplot(x="Rating", y="Broad Bean\nOrigin", data=df)
b = data.groupby(['Company\nLocation'])['Rating'].median()
b = b.sort_values(ascending=False)
df = pd.DataFrame({'Company\nLocation':b.index,'Rating':b.values})
df.head()
plt.figure(figsize = (50,6))

ax = sns.barplot(x="Company\nLocation", y="Rating", data=df)
c = data.groupby(['Company\nLocation'])['coco % as num'].median()

c = c.sort_values(ascending = False)

c1 = pd.DataFrame({'Company\nLocation':c.index,'Median-Coco%':c.values})

c1.head()
c1.index = c1['Company\nLocation']

df.index = df['Company\nLocation']

#median_company = c1.join(df,how = 'left',lsuffix = '%Coco',rsuffix = '_medianRating')

median_company = c1.merge(df,on = 'Company\nLocation',how = 'left')
median_company
a = data.groupby(['Company\nLocation','Broad Bean\nOrigin'])['Rating'].median()

#a = a.sort_values(ascending = False)

a