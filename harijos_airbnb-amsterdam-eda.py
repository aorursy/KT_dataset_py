import pandas as pd

df = pd.read_csv("../input/airbnb-amsterdam/listings.csv")
df.hist(figsize= (16,9))
df.info()
df['reviews_per_month'].fillna(0,inplace=True)

del df['neighbourhood_group']

df['name'].fillna('Empty',inplace=True)

df['host_name'].fillna('Empty',inplace=True)

df[df.last_review.isnull()==False]

df['last_review'].fillna('Empty',inplace=True)
df.head()
import seaborn as sns

sns.heatmap(df.corr())
del df['reviews_per_month']
df.info()
categorical_columns = [c for c in df.columns 

                       if df[c].dtype.name == 'object']

numerical_columns = [c for c in df.columns 

                     if df[c].dtype.name != 'object']

print('categorical: ',categorical_columns)

print('numerical: ',numerical_columns)

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')

#cmap = sns.cubehelix_palette(as_cmap=True)

sns.scatterplot(x='number_of_reviews',y='price',data=df,alpha=0.5,\

                hue='room_type',\

                #palette=cmap,\

                legend="full")



plt.ylim(0, 2000)

#plt.xlim(-10, 400)

figure(num=None, figsize=(25, 12), dpi=80, facecolor='w', edgecolor='k')

#cmap = sns.cubehelix_palette(as_cmap=True)

sns.violinplot(y='price',x='neighbourhood',data=df[df.price < df['price'].quantile(.98)],)



plt.xticks(rotation=90)
sns.barplot(x='price', y = 'neighbourhood',data=df.groupby('neighbourhood').mean()['price'].unstack())
df.groupby('neighbourhood').mean()['price'].unstack()