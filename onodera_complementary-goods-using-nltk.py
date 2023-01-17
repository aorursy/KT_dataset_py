import pandas as pd

from glob import glob

import nltk

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



files = sorted(glob('../input/*'))

files
order_products = pd.concat([pd.read_csv('../input/order_products__prior.csv'), 

                            pd.read_csv('../input/order_products__train.csv')], 

                           ignore_index=1)
order_products.sort_values(['order_id', 'add_to_cart_order'], inplace=1)

order_products.reset_index(drop=1, inplace=1)

order_products.head(9)
products = pd.read_csv('../input/products.csv')

products.product_name = products.product_name.str.replace(' ', '-')

products.head()
order_products = pd.merge(order_products, products, on='product_id', how='left')
products[~products.product_id.isin(order_products.product_id)]
plt.figure(figsize=(15, 15))

sns.barplot(x='product_name', y='index', data=order_products.product_name.value_counts().reset_index().head(30),

           label='product_name')

plt.subplots_adjust(left=.4, right=.9)
n = 99999 # increase this if you like

order_tbl = order_products.head(n).groupby('order_id').product_name.apply(list)

bigrams = [list(nltk.bigrams(s)) for s in order_tbl]

cfd = nltk.ConditionalFreqDist(sum(bigrams, []))
cfd['Lemons']
comp_goods = []

for key in cfd.keys():

    for k,v in cfd[key].items():

        comp_goods.append([key+' and '+k, v])



comp_goods = pd.DataFrame(comp_goods, columns=['goods', 'cnt'])

comp_goods.sort_values('cnt', ascending=0).head()
plt.figure(figsize=(15, 15))

sns.barplot(x='cnt', y='goods', data=comp_goods.sort_values('cnt', ascending=0).head(30),

            label='cnt')

plt.subplots_adjust(left=.4, right=.9)