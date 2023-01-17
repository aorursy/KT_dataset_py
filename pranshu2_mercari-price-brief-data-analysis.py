import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_table('../input/train.tsv', engine='c')

print("(Rows, Columns) : ",train.shape)
train.head()
for col in train:

    val=train[col].isnull().sum()*100.0/train.shape[0]

    if val>0.0:

        print("Missing values in",col,":",val)
price = train['price']

print(price.describe())

plt.figure(figsize=(12,12))

plt.scatter(range(train.shape[0]),np.sort(price.values))

plt.ylabel('Price', fontsize=12)

plt.xlabel('Index', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

sns.countplot(x="item_condition_id", data=train, palette="Greens_d")

plt.xlabel('Condition', fontsize=12)

plt.ylabel('Frequency', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

sns.boxplot(x='item_condition_id', y='price', data=train[train['price']<100])

plt.ylabel('Price', fontsize=12)

plt.xlabel('Item Condition', fontsize=12)

plt.show()
plt.figure(figsize=(20, 15))

plt.hist(train[train['shipping']==1]['price'],normed=True, bins=100, range=[1,250],alpha=0.6,color=['crimson'])

plt.hist(train[train['shipping']==0]['price'], normed=True,alpha=0.6,bins=100, range=[1,250],color=['blue'])

plt.xlabel('Price', fontsize=15)

plt.ylabel('Frequency', fontsize=15)

plt.show()
brand = train['brand_name']

brand.describe()

from collections import Counter

c = Counter(brand.dropna())

print(c.most_common(10))
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train['name']))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width=1440, height=1080).generate(" ".join(str(v) for v in train['item_description']))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
def cat_name(cat):

    try:

        return cat.split('/')

    except:

        return np.nan, np.nan, np.nan



train['main'], train['l1'], train['l2'] = zip(*train['category_name'].apply(cat_name))
plt.figure(figsize=(12,12))

sns.boxplot(x='main', y='price', data=train[train['price']<100])

plt.ylabel('Price', fontsize=12)

plt.xlabel('Main Category', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train.groupby(["main"])["main"].count().reset_index(name='count').sort_values(['count'], ascending=False).head(10)
cat_count = train.groupby(["main","l1"])["l1"].count().reset_index(name='count')

cat_count.sort_values(['count'], ascending=False).head(10)
prices = train.groupby(['main','l1'])['price'].mean().reset_index(name='mean')

prices.sort_values(['mean'], ascending=False).head(10)
prices.sort_values(['mean']).head(10)
price_count = prices.merge(cat_count,left_on=['main','l1'],  right_on = ['main','l1'],how='inner')

price_count = price_count.sort_values(['count'], ascending=False).head(50)

plt.figure(figsize=(20, 15))

plt.barh(range(0,len(price_count)), price_count['mean'], align='center', alpha=0.5)

plt.xlabel('Price', fontsize=15)

plt.yticks(range(0,len(price_count)), price_count.l1, fontsize=15)

plt.ylabel('Level 1 category', fontsize=15)
prices = train.groupby(['main','l1','l2'])['price'].mean().reset_index(name='mean')

prices.sort_values(['mean'], ascending=False).head(10)