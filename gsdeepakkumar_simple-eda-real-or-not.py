import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

import gc



from wordcloud import STOPWORDS

import string

import seaborn as sns

!ls ../input
DATA_PATH='../input/nlp-getting-started'
train=pd.read_csv(f'{DATA_PATH}/train.csv')

test=pd.read_csv(f'{DATA_PATH}/test.csv')
train.head()
print(f'Train data has {train.shape[0]} rows')

print(f'Test data has {test.shape[0]} rows')
train['keyword'].isna().sum()
print(f'The percentage of NA values in the keyword column {round((train.keyword.isna().sum()/train.shape[0])*100,2)} %')
print(f'There are {train.keyword.nunique()} unique keyword in the column')
real_keywords=train.loc[train['target']==1]['keyword'].dropna().unique()

nonreal_keywords=train.loc[train['target']==0]['keyword'].dropna().unique()
print(f'There are {len(real_keywords)} real keywords and {len(nonreal_keywords)} non real keywords in the train dataset')
print(f'Unique keywords found in real tweets only {set(real_keywords)-set(nonreal_keywords)}')

print(f'Unique keywords found in non real tweets only {set(nonreal_keywords)-set(real_keywords)}')
train.loc[train['keyword'].str.startswith('derail',na=False)]['keyword'].unique()
train.loc[train['keyword'].str.startswith('wreck',na=False)]['keyword'].unique()
train[train.target==1]['keyword'].dropna().value_counts()[0:5]
train[train.target==0]['keyword'].dropna().value_counts()[0:5]
## Creating the features:



train['n_words']=train['text'].apply(lambda x:len(str(x).split()))

train['n_unique_words']=train['text'].apply(lambda x:len(set(str(x).split())))

train['n_characters']=train['text'].apply(lambda x:len(str(x)))

train['n_stopwords']=train['text'].apply(lambda x:len([w for w in str(x).lower().split() if w in STOPWORDS]))

train['n_punctuations']=train['text'].apply(lambda x:len([w for w in str(x) if w in string.punctuation ]))

train['n_avg_words']=train['text'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))
train.head()
columns=['n_words','n_unique_words','n_characters','n_stopwords','n_punctuations','n_avg_words']
for c in columns:

    plt.figure(figsize=(8,8))

    ax=sns.boxplot(x='target',y=c,data=train)

    ax.set_xlabel(xlabel='Target')

    ax.set_ylabel(ylabel=c)

    plt.title(r'Boxplot of {} vs Target'.format(c))

    plt.show()