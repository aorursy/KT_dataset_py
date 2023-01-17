# Load in the CSV files and do some clean up
import pandas as pd 

biz = pd.read_csv('../input/yelp-dataset/yelp_business.csv')
tags = pd.read_csv('../input/yelp-categories/yelp_health_categories.clean.csv')

biz['categories'] = biz['categories'].str.split(';')
biz.head()
# Filter down to the Health businesses only
mask = biz['categories'].apply(lambda x: tags['Title'].isin(x).any())

df = biz[mask]
df
import seaborn as sns
sns.distplot(biz['stars'], norm_hist=True, kde=False)
sns.distplot(df['stars'], norm_hist=True, kde=False)
print('Only Health related Yelp entries')
df[['stars', 'review_count']].describe()
print('All Yelp businesses')
biz[['stars', 'review_count']].describe()
# biz['categories'].apply(lambda x: tags['Title'][tags['Title'].isin(x)])

from collections import Counter
flat = df['categories'].apply(Counter)

flat
from functools import reduce

counts = reduce(lambda x, y: x + y, flat)
s = pd.Series(dict(counts))

mask = s.index.isin(tags['Title'])
s = s[mask]


n = 100
pd.set_option('display.max_rows', n)
s.sort_values(ascending=False).head(n)
