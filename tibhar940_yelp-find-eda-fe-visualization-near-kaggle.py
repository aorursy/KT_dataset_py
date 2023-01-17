## imports and settings

import os

import re

import math

import itertools

import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import plot, iplot, init_notebook_mode

import networkx as nx

import warnings

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

pd.set_option('precision', 4)

warnings.simplefilter('ignore')

init_notebook_mode()



%matplotlib inline
## Explore input foler

os.listdir('../input')
%%time

## Load data

user = pd.read_csv('../input/yelp_user.csv')

business = pd.read_csv('../input/yelp_business.csv')

business_hours = pd.read_csv('../input/yelp_business_hours.csv', na_values='None')

business_attributes = pd.read_csv('../input/yelp_business_attributes.csv')

checkin = pd.read_csv('../input/yelp_checkin.csv')

tip = pd.read_csv('../input/yelp_tip.csv')

review = pd.read_csv('../input/yelp_review.csv')



## Little types transform

user['yelping_since'] = pd.to_datetime(user['yelping_since'])
## Define dataset names

dnames = ['user', 'business', 'business_hours', 'business_attributes', 'checkin', 'tip', 'review']

## Explore shapes of datasets

for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):

    print(n, d.shape)
## First look at datasets

for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):

    ## business_attrubutes is too many columns

    print('---------{0}---------'.format(n))

    if n != 'business_attributes':

        print(d.head(1).T)

    else:

        print(d.columns)
## Find keys in datasets

colnames = []

for d in [user, business, business_hours, business_attributes, checkin, tip, review]:

    colnames.extend(d.columns)

colnames = pd.Series(colnames).value_counts().reset_index()

colnames.columns = ['colname', 'cnt']

colnames[colnames['cnt'] > 1]
G = nx.Graph()

fig, ax = plt.subplots(figsize=[7,7])

for n, d in zip(dnames, [user, business, business_hours, business_attributes, checkin, tip, review]):

    _ = []

    for c in np.intersect1d(d.columns, ['business_id', 'user_id']):

        _.append([n, c])

    G.add_edges_from(_, label=n)

nx.draw_networkx(G, ax=ax)

plt.show()
## How many users make reviews?

print('Total users:', user['user_id'].nunique())

## How many user_id's contains in reviews

print('Total users review:', review['user_id'].nunique())

## Is there any different user_id?

print('Different user_id:', np.setdiff1d(review['user_id'], user['user_id']),

      review['user_id'].nunique()/user['user_id'].nunique())

## How many observations of this different user_id?

print('Different user_id shape:', 

      review[review['user_id'].isin(np.setdiff1d(review['user_id'], user['user_id']))].shape)

## How many user_id's contains in tip

print('Total users tip:', tip['user_id'].nunique())

## Is there any different user_id?

print('Tip users - users:', len(np.setdiff1d(tip['user_id'], user['user_id'])),

      tip['user_id'].nunique()/user['user_id'].nunique())

print('Users - tip users:', len(np.setdiff1d(user['user_id'], tip['user_id'])))
## How many organization contains our dataset?

print('Total organization:', business['business_id'].nunique())

## How many organizations has reviews?

print('Total organizations with review:', review['business_id'].nunique(),

      review['business_id'].nunique()/business['business_id'].nunique())

## How many organizations has tip?

print('Total organization with tip:', tip['business_id'].nunique(),

      tip['business_id'].nunique()/business['business_id'].nunique())

## How many organization have buisiness_hours info?

print('Total organizations with buisiness hours info:', business_hours['business_id'].nunique(),

      business_hours['business_id'].nunique()/business['business_id'].nunique())

## How many organization have buisiness_attributes info?

print('Total organizations with buisiness attributes info:',business_attributes['business_id'].nunique(),

      business_attributes['business_id'].nunique()/business['business_id'].nunique())

## How many organization have checkin info?

print('Total organizations with checkin info:', checkin['business_id'].nunique(),

      checkin['business_id'].nunique()/business['business_id'].nunique())
## Look at the most active users

user.sort_values('review_count', ascending=False).head(2).T
user.describe(include='all').T
#### Look at distribution for numeric variables

user_desc = user.dtypes.reset_index()

user_desc.columns = ['variable', 'type']

cols = user_desc[user_desc['type']=='int64']['variable']

fig, ax = plt.subplots(math.ceil(len(cols)/2), 2, figsize=[12, math.ceil(len(cols)/2)*2])

ax = list(itertools.chain.from_iterable(ax))

for ax_, v in zip(ax[:len(cols)], cols):

    sns.distplot(np.log1p(user[v]), ax=ax_, label=v)

    ax_.set_xticklabels(np.expm1(ax_.get_xticks()).round())

    ax_.legend()

plt.show()
business.head(2).T
business.describe(include='all').T
#### Look at distribution for numeric variables

business_desc = business.dtypes.reset_index()

business_desc.columns = ['variable', 'type']

cols = business_desc[business_desc['type']=='int64']['variable']

fig, ax = plt.subplots(math.ceil(len(cols)/2), 2, figsize=[12, math.ceil(len(cols)/2)*2])

for ax_, v in zip(ax[:len(cols)], cols):

    sns.distplot(np.log1p(business[v]), ax=ax_, label=v)

    ax_.set_xticklabels(np.expm1(ax_.get_xticks()).round())

    ax_.legend()

plt.show()
## Look at count for categorical variables

cols = ['neighborhood', 'city', 'state']

for c in cols:

    print(business[c].value_counts(normalize=True).head())
## What about categories of organizations

## How many categories in each organization? (minuimum 1, maximum 35 categories)

## Most frequent 2 categories

print(business['categories'].str.count(';').min() + 1, business['categories'].str.count(';').max() + 1)

(business['categories'].str.count(';') + 1).value_counts().head()
## How many categories we have?

categories = pd.concat(

    [pd.Series(row['business_id'], row['categories'].split(';')) for _, row in business.iterrows()]

).reset_index()

categories.columns = ['categorie', 'business_id']

categories.head()
## How many categories?

print(categories['categorie'].nunique())

## Most frequent categories

categories['categorie'].value_counts().head(10)
fig, ax = plt.subplots(figsize=[5,10])

sns.countplot(data=categories[categories['categorie'].isin(

    categories['categorie'].value_counts().head(25).index)],

                              y='categorie', ax=ax)

plt.show()
categories_ = categories[

    (categories['categorie'].isin(categories['categorie'].value_counts().head(25).index))

]

ct = pd.crosstab(

    categories_['business_id'],

    categories_['categorie'])



fig, ax = plt.subplots(figsize=[10,10])

sns.heatmap(ct.head(25), ax=ax, cmap='Reds')

ax.set_title('Top 25-cat, Random 25 organizations')
## Also we have geospatial data

g = sns.jointplot(data=business, x='longitude', y='latitude', size=8, stat_func=None)
review.head(2).T
review.describe(include='all').T
tip.head(2).T
tip.describe(include='all').T
business_hours.head()
business_hours.describe(include='all').T
checkin.head()
checkin.describe(include='all').T
business_attributes.columns
## function for get time_range from string

def get_time_range(s):

    if isinstance(s, str):

        t1, t2 = s.split('-')

        h1, m1 = map(int, t1.split(':'))

        h2, m2 = map(int, t2.split(':'))

        m1, m2 = m1/60, m2/60

        t1, t2 = h1+m1, h2+m2

        if t2 < t1:

            d = t2+24-t1

        else:

            d = t2-t1

        return t1, t2, d

    else:

        return None, None, None
%%time

## Prepare start/finish/delta features for every weekday

bh_colnames = business_hours.columns

for c in bh_colnames[1:]:

    business_hours['{0}_s'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[0])

    business_hours['{0}_f'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[1])

    business_hours['{0}_d'.format(c[:2])] = business_hours[c].apply(lambda d: get_time_range(d)[2])

business_hours = business_hours.drop(bh_colnames[1:], axis=1)
business_hours.head()
## Look at our features

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[15, 4])

sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_s' in c]].corr(),

            cmap='Reds', ax=ax1)

sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_f' in c]].corr(),

            cmap='Greens', ax=ax2)

sns.heatmap(business_hours.loc[:, [c for c in business_hours.columns if '_d' in c]].corr(),

            cmap='Blues', ax=ax3)

ax1.set_title('Start hours heatmap')

ax2.set_title('Finish hours heatmap')

ax3.set_title('Duration heatmap')

plt.show()
## Look at our features

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=[15, 9])

for wd in [c for c in business_hours.columns if '_s' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax1, label=wd)

for wd in [c for c in business_hours.columns if '_f' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax2, label=wd)

for wd in [c for c in business_hours.columns if '_d' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax3, label=wd)

ax1.legend()

ax2.legend()

ax3.legend()

ax1.set_title('Start hours distribution')

ax2.set_title('Finish hours distribution')

ax3.set_title('Duration distribution')

plt.show()
wd = ['mo', 'tu', 'we', 'th']

fr = ['fr']

ho = ['sa', 'su']



## define new_cols

bh_newcols = ['business_id']

for wg_name, wg in zip(['wd', 'fr', 'ho'], [wd, fr, ho]):

    for f in ['s', 'f', 'd']:

        cols = list(map(lambda d: '{0}_{1}'.format(d,f), wg))

        bh_newcols.append('{0}_{1}'.format(wg_name, f))

        business_hours['{0}_{1}'.format(wg_name, f)] = business_hours.loc[:, cols].median(axis=1)



business_hours.loc[:, bh_newcols].head()
## Look at our new features distribution

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=[15, 9])

for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_s' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax1, label=wd)

for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_f' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax2, label=wd)

for wd in [c for c in business_hours.loc[:, bh_newcols].columns if '_d' in c]:  

    sns.distplot(business_hours[wd].dropna(), ax=ax3, label=wd)

ax1.legend()

ax2.legend()

ax3.legend()

ax1.set_title('Start hours distribution')

ax2.set_title('Finish hours distribution')

ax3.set_title('Duration distribution')

plt.show()
## Join our new features to business dataframe

business = business.merge(business_hours.loc[:, bh_newcols])

business.head(2).T