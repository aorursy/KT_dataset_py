import json

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.io.json import json_normalize



pd.pandas.set_option('display.max_columns',None)
## import and normalize json data to get categories for each id

with open('../input/youtube-new/US_category_id.json') as f:

    d = json.load(f)



df_cat = pd.json_normalize(d['items'])[['id','snippet.title']]

df = pd.read_csv('../input/youtube-new/USvideos.csv')



## change dtype and prepare to join data on id

df['category_id'] = df['category_id'].astype(int)

df_cat['id'] = df_cat['id'].astype(int)

df = df.join(df_cat.set_index('id'), on='category_id')



## rename and drop unnecessary column

df.rename(columns={'snippet.title':'category'}, inplace=True)

df.drop('category_id', axis=1, inplace=True)
print(f'DF Shape: {df.shape} \n')

print(df.nunique())
df.head(3)
df.describe()
features_nan = [f for f in df.columns if df[f].isnull().sum() > 1]

for feature in features_nan:

    print('Feature:',feature, np.round(df[feature].isnull().mean()*100,2),'% of values missing')
num_feat = [f for f in df.columns if df[f].dtype in ['int64','float64']]

cat_feat = [f for f in df.columns if df[f].dtype in ['object','bool']]



print(f'Number of numerical variables: {len(num_feat)}')

print(f'Number of categorical variables: {len(cat_feat)}')
## check for skewness since outliers are present as seen in general overview

plt.figure(figsize=(14,6))



for idx,feature in enumerate(num_feat):

    print('Skewness of',feature, df[feature].skew())



    color = 'lightblue'

    if 'likes' in feature: color = 'green'

    if 'dislike' in feature: color = 'red'



    plt.subplot(2,2,idx+1)

    g = sns.distplot(df[feature], color=color)

    g.set_title(f'{feature} Distribution', fontsize=14)



plt.subplots_adjust(wspace=.2, hspace=.4, top=.9)

plt.show()
## log transform to get rid of highly skewed distribution

## log transformation and obtained "normal" distribution allows us to better analyze correlation

for feature in num_feat:

    df[feature+'_log'] = np.log(df[feature]+1)

    

num_log_feat = [f for f in df.columns if '_log' in f]
sns.set_style('dark')

plt.figure(figsize=(14,6))



for idx,feature in enumerate(num_log_feat):

    color = 'lightblue'

    if 'likes' in feature: color = 'green'

    if 'dislike' in feature: color = 'red'



    plt.subplot(2,2,idx+1)

    g = sns.distplot(df[feature], color=color)

    g.set_title(f'{feature} Distribution', fontsize=14)



plt.subplots_adjust(wspace=.2, hspace=.4, top=.9)

plt.show()
## check correlation matrix for numerical variables

plt.figure(figsize=(14,9))

sns.heatmap(df[num_log_feat].corr(), linewidths=.5, annot=True, cbar=False, cmap='Blues');
print('Top 5 Category Count:')

print(df['category'].value_counts()[:5])
plt.figure(figsize=(18,24))



plt.subplot(5,1,1)

g1 = sns.countplot(df['category'], data=df, palette='pastel')

g1.set_xticklabels(g.get_xticklabels(), rotation=45)

g1.set_xlabel('',fontsize=12)

g1.set_ylabel('Count', fontsize=12)

g1.set_title('No. of Videos per Category', fontsize=12, weight='bold')



for idx,feature in enumerate(num_log_feat):

    plt.subplot(5,1,idx+2)

    g = sns.boxplot(x='category', y=feature, data=df, palette='GnBu_d')

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_xlabel('', fontsize=12)

    g.set_ylabel(feature, fontsize=12)

    g.set_title(f'{feature} Distribution by Category', fontsize=12, weight='bold')



plt.subplots_adjust(hspace=0.6)

plt.show()
## convert publish time into datetime

df['publish_time'] = pd.to_datetime(df['publish_time'])

df['publish_year'] = df['publish_time'].dt.year

df['publish_month'] = df['publish_time'].dt.month

df['publish_day'] = df['publish_time'].dt.day_name()

df['publish_hour'] = df['publish_time'].dt.hour
## datetime features

publish_feat = [f for f in df.columns if 'publish' in f and not 'publish_time' in f]



plt.figure(figsize=(18,24))

rows = len(publish_feat)



for idx,feature in enumerate(publish_feat):

    plt.subplot(rows,1,idx+1)

    g = sns.boxplot(x=feature, y='views_log', data=df)

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_xlabel('', fontsize=12)

    g.set_ylabel('views(log)', fontsize=12)

    g.set_title(f'Views per {feature}', fontsize=12, weight='bold')



plt.subplots_adjust(hspace=0.9)

plt.show()
## create features based on the numerical variables in relation to the overall views

eng_feat = df[['likes','dislikes','comment_count']]



for feature in eng_feat:

    df[feature+'_rate'] = df[feature] / df['views'] * 100
eng_rate_feat = [f for f in df.columns if '_rate' in f]

rows = len(eng_rate_feat)



plt.figure(figsize=(12,18))



for idx, feature in enumerate(eng_rate_feat):

    plt.subplot(rows, 1, idx + 1)

    g = sns.boxplot(x='category', y=feature, data=df)

    g.set_xticklabels(g.get_xticklabels(), rotation=45)

    g.set_xlabel('', fontsize=12)

    g.set_ylabel(feature, fontsize=12)

    g.set_title(f'{feature} by Category', fontsize=12, weight='bold')



plt.subplots_adjust(hspace=0.5)

plt.show()