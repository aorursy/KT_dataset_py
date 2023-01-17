# This notebook explores the character distribution of description.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot
train = pd.read_csv('../input/train.csv', usecols=['description', 'deal_probability'])
test = pd.read_csv('../input/test.csv', usecols=['description'])

df = pd.concat((train, test))

df.index = range(df.shape[0])
charvec = CountVectorizer(
    analyzer='char',
    lowercase=False,
    max_df=1.0,
    min_df=1
)
char_counts = charvec.fit_transform(df['description'].fillna(''))
char_counts
totals = pd.DataFrame(
    np.array(char_counts.sum(axis=0))[0], 
    index=charvec.get_feature_names(),
    columns=['cnt']
)
totals['ord'] = totals.index.map(lambda x: ord(x))
totals['cat'] = totals.index.map(lambda x: unicodedata.category(x))
def extract_name(x):
    try:
        if '\t' == x:
            return 'CHARACTER TABULATION'
        if '\n' == x:
            return 'LINE FEED'
        return unicodedata.name(x)
    except:
        return None
    
totals['name'] = totals.index.map(extract_name)
totals['name'].fillna('', inplace=True)
r = totals.groupby('cat').cnt.agg(['count', 'sum'])
r.sort_values('sum', ascending=False)
(r / r.sum()).sort_values('sum', ascending=False).plot(kind='bar', figsize=(12, 4))
charset_idx = np.array(range(totals.shape[0]))[totals['cat'] == 'Po']
df['charset_cnt'] = np.log2(char_counts[:, charset_idx].sum(axis=1) + 1).astype(int)
df.groupby('charset_cnt').deal_probability.mean().plot(kind='bar', color='#7777ac')
del df['charset_cnt']
for cat in totals['cat'].unique():
    print(cat)
    feature = 'charset_{}_cnt'.format(cat)
    charset_idx = np.array(range(totals.shape[0]))[totals['cat'] == cat]
    df[feature] = np.log2(char_counts[:, charset_idx].sum(axis=1) + 1).astype(int)
nu_cats = totals.cat.nunique()
nu_cats
charset_cols = list(filter(lambda x: x.startswith('charset_'), df.columns))
max_vals = df[charset_cols].max().sort_values(ascending=False)
max_vals[max_vals <= 8].index.shape
f, axes = pyplot.subplots(3, 3, sharey=True, figsize=(15, 10))
axes = axes.flatten()
for k, feat in enumerate(max_vals[max_vals > 8].index):
    r = df.groupby(feat).deal_probability.agg(['count', 'mean'])
    r['pcnt_cnt'] = r['count'] / r['count'].sum() 
    r[['pcnt_cnt', 'mean']].plot(kind='bar', color=['#667799', '#aa3366'], ax=axes[k], title=feat)
    axes[k].set_xlabel('')
f, axes = pyplot.subplots(3, 6, sharey=True, figsize=(15, 10))
axes = axes.flatten()
for k, feat in enumerate(max_vals[max_vals <= 8].index):
    r = df.groupby(feat).deal_probability.agg(['count', 'mean'])
    r['pcnt_cnt'] = r['count'] / r['count'].sum() 
    r[['pcnt_cnt', 'mean']].plot(kind='bar', color=['#667799', '#aa3366'], ax=axes[k], title=feat)
    axes[k].set_xlabel('')
totals['idx'] = range(totals.shape[0])
totals.loc['!']
df['!_cnt'] = char_counts[:, 3]

