# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

df=pd.read_csv('../input/flipkart-products/flipkart_com-ecommerce_sample.csv')
df.head()
plt.figure(figsize=(15,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

plt.show()
df.duplicated().value_counts()
df['crawl_timestamp']=pd.to_datetime(df['crawl_timestamp'])
df['crawl_year']=df['crawl_timestamp'].apply(lambda x :x.year)

df['crawl_month']=df['crawl_timestamp'].apply(lambda x :x.month)
print(df.product_category_tree[1])
df.product_category_tree[6].split('>>')[0][2:].strip('[]"').strip()
df['MainCategory'] = df['product_category_tree'].apply(lambda x: x.split('>>')[0][2:].strip('[]"').strip())
df['MainCategory'].head()
plt.figure(figsize=(10,10))

print(df.groupby('crawl_month')['crawl_month'].count())

df.groupby('crawl_month')['crawl_month'].count().plot(kind='bar')

plt.title('Sales count by Month')

plt.xlabel('Month')

plt.ylabel('Count')

plt.show()

plt.figure(figsize=(5,5))

df.groupby('crawl_year')['crawl_year'].count().plot(kind='bar')

plt.title('Sales by Year')

plt.xlabel('Year')

plt.ylabel('Count')
plt.figure(figsize=(12,8))

df.groupby('MainCategory')['MainCategory'].count().sort_values(ascending=False)[:15]

df['MainCategory'].value_counts()[:15].sort_values(ascending=True).plot(kind='barh')

df['retail_price'].max()
df[df['retail_price']==571230.0]
df['discount_%']=round(((df['retail_price']-df['discounted_price'])/df['retail_price'])*100,1)
main_cat_disc=pd.DataFrame(df.groupby('MainCategory').agg({'discount_%':[(np.mean)],

                                                          'MainCategory':['count']}))
main_cat_disc.head()

main_cat_disc.columns=['_'.join(col) for col in main_cat_disc.columns]
MainCategoryDiscount = main_cat_disc.sort_values(by=['MainCategory_count'],ascending=False)[:20]
MainCategoryDiscount
plt.figure(figsize=(12,8))

MainCategoryDiscount['discount_%_mean'].sort_values(ascending=True)[:20].plot(kind='barh')