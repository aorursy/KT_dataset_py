import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack
from scipy.stats import probplot
import pickle
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
import seaborn as sns 
import gc
import warnings
warnings.filterwarnings('ignore')
import time

color = sns.color_palette()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.palplot(color)

import os
PATH = "../input"
!du -l ../input/*
def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result

from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm_notebook
     
def process_line(line):
    json_data = read_json_line(line)
    content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
    content_no_html_tags = strip_tags(content)
    published = json_data['published']['$date']
    title = json_data['meta_tags']['title'].split('\u2013')[0].strip() #'Medium Terms of Service – Medium Policy – Medium'
    author = json_data['meta_tags']['author'].strip()
    domain = json_data['domain']
    url = json_data['url']

    tags_str = []
    soup = BeautifulSoup(content, 'lxml')
    try:
        tag_block = soup.find('ul', class_='tags')
        tags = tag_block.find_all('a')
        for tag in tags:
            tags_str.append(tag.text.translate({ord(' '):None, ord('-'):None}))
        tags = ' '.join(tags_str)
    except Exception:
        tags = 'None'
    return content_no_html_tags, published, title, author, domain, tags, url
 
def extract_features(path_to_data):
    
    content_list = [] 
    published_list = [] 
    title_list = []
    author_list = []
    domain_list = []
    tags_list = []
    url_list = []

    with open(path_to_data, encoding='utf-8') as inp_json_file:
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(process_line)(line) for line in tqdm_notebook(inp_json_file))
    return zip(*results) #content_list, published_list, title_list, author_list, domain_list, tags_list, url_list
content_train, published_list, title_list, author_list, domain_list, tags_list, url_list = extract_features(os.path.join(PATH, 'how-good-is-your-medium-article/train.json'))
train = pd.DataFrame()
#train['content'] = content_list
train['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
train['title'] = title_list
train['author'] = author_list
train['domain'] = domain_list
train['tags'] = tags_list
train['length'] = list(map(len, content_train))
train['url'] = url_list

train_target = pd.read_csv(os.path.join(PATH, 'how-good-is-your-medium-article/train_log1p_recommends.csv'), index_col='id')
y_train = train_target['log_recommends'].values
content_test, published_list, title_list, author_list, domain_list, tags_list, url_list = extract_features(os.path.join(PATH, 'how-good-is-your-medium-article/test.json'))
test = pd.DataFrame()
#test['content'] = content_list
test['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
test['title'] = title_list
test['author'] = author_list
test['domain'] = domain_list
test['tags'] = tags_list
test['length'] = list(map(len, content_test))
test['url'] = url_list
del published_list, title_list, author_list, domain_list, tags_list, url_list
gc.collect()
idx_split = len(train)
df_full = pd.concat([train, test])

df_full['dow'] = df_full['published'].apply(lambda x: x.dayofweek)
df_full['year'] = df_full['published'].apply(lambda x: x.year)
df_full['month'] = df_full['published'].apply(lambda x: x.month)
df_full['hour'] = df_full['published'].apply(lambda x: x.hour)
df_full['number_of_tags'] = df_full['tags'].apply(lambda x: len(x.split()))

train = df_full.iloc[:idx_split, :]
test = df_full.iloc[idx_split:, :]

train['target'] = y_train
train.sort_values(by='published', inplace=True)
train.reset_index(drop=True, inplace=True)

print('TRAIN: {}'.format(train.shape))
print('TEST: {}'.format(test.shape))
del df_full
gc.collect()
train.head()
plt.figure(figsize=(15,6))
plt.suptitle("Target variable",fontsize=20)
gridspec.GridSpec(2,2)

plt.subplot2grid((2,2),(0,0))
plt.xlim(0, 12)
sns.distplot(train.target.values, hist=False, color=color[0], kde_kws={"shade": True, "lw": 2})
plt.title("Number of claps (log1p transformed)")

plt.subplot2grid((2,2),(1,0))
plt.xlim(0, 12)
sns.boxplot(train.target.values)

plt.subplot2grid((2,2),(0,1), rowspan=2)
plt.ylim(0, 12)
# plt.grid(False)
probplot(train.target.values, dist="norm", plot=plt);
train.sort_values(by='target', ascending=False).reset_index(drop=True).loc[0, 'url']
plt.figure(figsize=(16,6))
plt.suptitle("                       Posts distribution across years",fontsize=20)

ax1 = plt.subplot2grid((1,5),(0,0), colspan=3)
ax1 = sns.countplot(x='year', data=train, alpha=0.8, color=color[2])
plt.ylabel('Overall posts', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.title('Train data', fontsize=15)
plt.grid(False)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom')
    
ax2 = plt.subplot2grid((1,5),(0,3), colspan=2, sharey=ax1)
ax2 = sns.countplot(x='year', data=test, alpha=0.8, color=color[9])
plt.xlabel('Year', fontsize=12)
plt.title('Test data', fontsize=15)
plt.yticks([])
plt.ylabel('')

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom')
train = train[train.year >= 2015]
temp=pd.concat([train.groupby(['year','month'])['hour'].count(), test.groupby(['year','month'])['hour'].count().iloc[:-1]])
plt.figure(figsize=(12,4))
sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[1],)
plt.ylabel('Overall posts', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.title('Monthly posts variation', fontsize=15)
plt.xticks(rotation='vertical');

temp=train.groupby(['year','month']).aggregate({'hour':np.size,'year':np.min,'month':np.min})
temp.reset_index(drop=True, inplace=True)
plt.figure(figsize=(12,6))
plt.plot(range(1,13),temp.iloc[0:12,0],label="2015", marker='o')
plt.plot(range(1,13),temp.iloc[12:24,0],label="2016", marker='o')
plt.plot(range(1,7),temp.iloc[24:30,0],label="2017-train", marker='o')
connect_point = temp.iloc[29,0]

temp=test.groupby(['year','month']).aggregate({'hour':np.size,'year':np.min,'month':np.min})
temp.reset_index(drop=True, inplace=True)
plt.plot(range(6,8),[connect_point,temp.iloc[0,0]], color='r',label=None)
plt.plot(range(7,13),temp.iloc[0:6,0], color='r',label="2017-test", marker='o')
plt.plot(range(1,3),temp.iloc[6:8,0],label="2018-test", marker='o')
plt.ylabel('Overall posts', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.title('Monthly posts variation', fontsize=15)
plt.xticks(np.arange(1, 13, 1.0))
plt.xlim(1, 12)
plt.legend(loc='upper right', fontsize=11)
plt.xticks(rotation='horizontal');
temp=train.groupby(['year','month'])['target'].sum()
plt.figure(figsize=(13,4))
sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[4],)
plt.ylabel('Overall claps (log1p transformed)', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.title('Monthly claps variation', fontsize=15)
plt.xticks(rotation='vertical');
plt.figure(figsize=(16,6))

plt.subplot(121)
ax1 = sns.boxplot(y='target',x='dow', data=train)
plt.ylabel('Claps by post', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title('Claps distribution across day of week', fontsize=15)
ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

plt.subplot(122)
temp = train.groupby('dow')['target'].sum()
ax2 = sns.barplot(temp.index,np.round(temp.values))
plt.ylabel('Number of claps', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title('Count of claps across day of week', fontsize=15)
ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(16,6))

plt.subplot(121)
ax1 = sns.boxplot(y='target',x='hour', data=train, color=color[9])
plt.ylabel('Claps by post', fontsize=12)
plt.xlabel('Hour', fontsize=12)
plt.title('Claps distribution across hour', fontsize=15)

plt.subplot(122)
temp = train.groupby('hour')['target'].sum()
ax2 = sns.barplot(temp.index,temp.values, alpha=0.8, color=color[9])
plt.ylabel('Number of claps', fontsize=12)
plt.xlabel('Hour', fontsize=12)
plt.title('Count of claps across hour', fontsize=15);
plt.figure(figsize=(15,10))
plt.title('Claps distribution across hour and day of week', fontsize=20)
temp = train.pivot_table(index='dow', columns='hour', values='target', aggfunc='mean')
ax = sns.heatmap(temp, annot=True, fmt='.2f', cmap='viridis')
ax.set(xlabel='Hour', ylabel='Day of week')
ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.yticks(rotation='horizontal');
plt.figure(figsize=(16,6))
plt.suptitle("Posts distribution across domains",fontsize=20)

ax1 = plt.subplot2grid((1,2),(0,0))
ax1 = sns.countplot(x='domain', data=train, alpha=0.8, color=color[2], order=train.domain.value_counts().iloc[:10].index)
plt.ylabel('Number of posts', fontsize=12)
plt.xlabel('Domain', fontsize=12)
plt.title('Train data', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom')
    
ax2 = plt.subplot2grid((1,2),(0,1), sharey=ax1)
ax2 = sns.countplot(x='domain', data=test, alpha=0.8, color=color[9], order=test.domain.value_counts().iloc[:10].index)
plt.xlabel('Domain', fontsize=12)
plt.title('Test data', fontsize=15)
plt.yticks([])
plt.ylabel('')
plt.xticks(rotation=90)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(16,6))

plt.subplot(121)
temp = train[train.domain.isin(['medium.com', 'hackernoon.com'])]
ax1 = sns.boxplot(y='target',x='year', hue='domain', data=temp)
plt.ylabel('Claps by post', fontsize=12)
plt.xlabel('Domain', fontsize=12)
plt.title('Claps distribution across domains', fontsize=15)

plt.subplot(122)
temp = temp.groupby('domain')['target'].sum().iloc[:2]
ax2 = sns.barplot(temp.index,temp.values)
plt.ylabel('Number of claps', fontsize=12)
plt.xlabel('Domain', fontsize=12)
plt.title('Count of claps across domain', fontsize=15);
plt.figure(figsize=(16,6))
plt.suptitle("Count of posts across authors",fontsize=20)

ax1 = plt.subplot2grid((1,2),(0,0))
ax1 = sns.countplot(x='author', data=train, alpha=0.8, color=color[2], order=train.author.value_counts().iloc[:10].index)
plt.ylabel('Overall posts', fontsize=12)
plt.xlabel('Author', fontsize=12)
plt.title('Train data', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom')
    
ax2 = plt.subplot2grid((1,2),(0,1), sharey=ax1)
ax2 = sns.countplot(x='author', data=test, alpha=0.8, color=color[9], order=test.author.value_counts().iloc[:10].index)
plt.xlabel('Author', fontsize=12)
plt.title('Test data', fontsize=15)
plt.yticks([])
plt.ylabel('')
plt.xticks(rotation=90, fontsize=12)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(18,6))


temp = train.groupby('author')['target'].sum().sort_values(ascending=False).iloc[:30]
ax1 = sns.barplot(temp.index,np.round(temp.values, 1), alpha=0.8, color=color[3])
plt.ylabel('Number of claps', fontsize=12)
plt.xlabel('Author', fontsize=12)
plt.title('Number of claps across authors', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom')
    
plt.figure(figsize=(18,6))


temp = train.groupby('author')['target'].median().sort_values(ascending=False).iloc[:30]
ax2 = sns.barplot(temp.index,np.round(temp.values, 1), alpha=0.8, color=color[4])
plt.ylabel('Median of claps by post', fontsize=12)
plt.xlabel('Author', fontsize=12)
plt.title('Median of claps across authors', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(15,6))
plt.suptitle("Length of post distribution",fontsize=20)
gridspec.GridSpec(2,1)

plt.subplot2grid((2,1),(0,0))
plt.xlim(0, 450000)
sns.distplot(train.length.values, hist=False, color=color[0], kde_kws={"shade": True, "lw": 2})
plt.title("Number of chars")

plt.subplot2grid((2,1),(1,0))
plt.xlim(0, 450000)
sns.boxplot(train.length.values);

plt.figure(figsize=(15,6))
plt.suptitle("Length of post distribution (log1p transformed)",fontsize=20)
gridspec.GridSpec(2,1)

plt.subplot2grid((2,1),(0,0))
sns.distplot(np.log1p(train.length.values), hist=False, color=color[0], kde_kws={"shade": True, "lw": 2})
plt.title("Number of chars (log1p transformed)")

plt.subplot2grid((2,1),(1,0))
sns.boxplot(np.log1p(train.length.values));
ax = sns.jointplot(x=np.log1p(train["length"]), y=train["target"], kind='kde', size=9)
ax.set_axis_labels("Length of article", "Number of claps");
%%time
cv_train_tags = CountVectorizer(ngram_range=(1, 1), min_df=5)
X_train_tags = cv_train_tags.fit_transform(train.tags.values).toarray()
cv_test_tags = CountVectorizer(ngram_range=(1, 1), min_df=5)
X_test_tags = cv_test_tags.fit_transform(test.tags.values).toarray()

matrix_freq = X_train_tags.sum(axis=0).ravel()
X_train_freq = np.array([np.array(cv_train_tags.get_feature_names()), matrix_freq])
matrix_freq = X_test_tags.sum(axis=0).ravel()
X_test_freq = np.array([np.array(cv_test_tags.get_feature_names()), matrix_freq])

df_train_tags = pd.DataFrame()
df_train_tags['tag'] = X_train_freq[0]
df_train_tags['number_of_posts'] = X_train_freq[1]
df_train_tags['mean_claps'] = [0]*len(X_train_freq[1])
df_train_tags['sum_claps'] = [0]*len(X_train_freq[1])

df_test_tags = pd.DataFrame()
df_test_tags['tag'] = X_test_freq[0]
df_test_tags['number_of_posts'] = X_test_freq[1]

df=pd.DataFrame(X_train_tags)
df['target'] = train.target.values
for col in range(df.shape[1]-1):
    temp=df[df[col]==1]
    df_train_tags.loc[col,'mean_claps']=temp['target'].mean()
    df_train_tags.loc[col,'sum_claps']=temp['target'].sum()
    
df_train_tags['tag'] = df_train_tags['tag'].astype(str)
df_train_tags['number_of_posts'] = df_train_tags['number_of_posts'].astype(int)
df_test_tags['tag'] = df_test_tags['tag'].astype(str)
df_test_tags['number_of_posts'] = df_test_tags['number_of_posts'].astype(int)
plt.figure(figsize=(16,6))
plt.suptitle("Top-15 tags by number of occurrences in posts", fontsize=18)

ax1 = plt.subplot2grid((1,2),(0,0))
temp = df_train_tags.sort_values(by='number_of_posts', ascending=False).iloc[:15]
ax1 = sns.barplot(temp.tag, temp.number_of_posts, alpha=0.8, color=color[7])
plt.ylabel('Number of occurrences', fontsize=12)
plt.xlabel('Tag', fontsize=12)
plt.title('Train data', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom')
    
ax2 = plt.subplot2grid((1,2),(0,1), sharey=ax1)
temp = df_test_tags.sort_values(by='number_of_posts', ascending=False).iloc[:15]
ax2 = sns.barplot(temp.tag, temp.number_of_posts, alpha=0.8, color=color[8])
plt.xlabel('Tag', fontsize=12)
plt.title('Test data', fontsize=15)
plt.yticks([])
plt.ylabel('')
plt.xticks(rotation=90, fontsize=12)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(18,6))
temp = df_train_tags.sort_values(by='sum_claps', ascending=False).iloc[:30]
ax1 = sns.barplot(temp.tag, temp.sum_claps, alpha=0.8, color=color[3])
plt.ylabel('Overall claps', fontsize=12)
plt.xlabel('Tag', fontsize=12)
plt.title('Top-30 tags by total number of claps', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom')
    
plt.figure(figsize=(18,6))
temp = df_train_tags.sort_values(by='mean_claps', ascending=False).iloc[:30]
ax2 = sns.barplot(temp.tag, temp.mean_claps, alpha=0.8, color=color[4])
plt.ylabel('Median of claps by post', fontsize=12)
plt.xlabel('Tag', fontsize=12)
plt.title('Top-30 tags by median of claps', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(np.round(p.get_height(),1)), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(16,6))

plt.subplot(121)
ax1 = sns.boxplot(y='target',x='number_of_tags', data=train)
plt.ylabel('Claps distribution', fontsize=12)
plt.xlabel('Number of tags in an article', fontsize=12)
plt.title('Claps distribution across article with different number of tags', fontsize=15)

plt.subplot(122)
temp = train.groupby('number_of_tags')['target'].sum()
ax2 = sns.barplot(temp.index,np.round(temp.values))
plt.ylabel('Number of claps', fontsize=12)
plt.xlabel('Number of tags in an article', fontsize=12)
plt.title('Count of claps across article with different number of tags', fontsize=15)

for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(int(p.get_height())), (x.mean(), y), ha='center', va='bottom');
plt.figure(figsize=(16,6))
plt.suptitle("Count of articles with different number of tags",fontsize=20)

ax1 = plt.subplot2grid((1,2),(0,0))
ax1 = sns.countplot(x='number_of_tags', data=train, alpha=0.8, color=color[1])
plt.ylabel('Overall articles', fontsize=12)
plt.xlabel('Number of tags in an article', fontsize=12)
plt.title('Train data', fontsize=15)
plt.grid(False)


for p in ax1.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax1.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom')
    
ax2 = plt.subplot2grid((1,2),(0,1), sharey=ax1)
ax2 = sns.countplot(x='number_of_tags', data=test, alpha=0.8, color=color[2])
plt.xlabel('Number of tags in an article', fontsize=12)
plt.title('Test data', fontsize=15)
plt.yticks([])
plt.ylabel('')


for p in ax2.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax2.annotate('{}'.format(p.get_height()), (x.mean(), y), ha='center', va='bottom');
content_train = train['content'].values.tolist()
title_train = train['title'].values.tolist()
tags_train = train['tags'].values.tolist()
y_train = train['target'].values
train.drop(['content', 'title', 'target', 'tags', 'published', 'length', 'url'], axis=1, inplace=True)

content_test = test['content'].values.tolist()
title_test = test['title'].values.tolist()
tags_test = test['tags'].values.tolist()
test.drop(['content', 'title', 'tags', 'published', 'length', 'url'], axis=1, inplace=True)
train.columns
test.columns
%%time

idx_split = len(train)
df_full = pd.concat([train, test])

list_to_dums = ['author', 'dow', 'month', 'hour', 'domain', 'year']
dummies = pd.get_dummies(df_full, columns = list_to_dums, drop_first=True,
                            prefix=list_to_dums, sparse=False)

X_train_feats = dummies.iloc[:idx_split, :]
X_test_feats = dummies.iloc[idx_split:, :]

print('TRAIN feats: {}'.format(X_train_feats.shape))
print('TEST feats: {}'.format(X_test_feats.shape))
del dummies, df_full
gc.collect()
%%time
# cv_title = CountVectorizer(max_features=30000)
cv_content = CountVectorizer(max_features=50000)
cv_tags = CountVectorizer(max_features=1000)

# X_train_title = cv_title.fit_transform(title_train)
# X_test_title = cv_title.transform(title_test)
X_train_content = cv_content.fit_transform(content_train)
X_test_content = cv_content.transform(content_test)
X_train_tags = cv_tags.fit_transform(tags_train)
X_test_tags = cv_tags.transform(tags_test)

print('TRAIN content: {}, tags: {}'.format(X_train_content.shape, X_train_tags.shape))
print('TEST content: {}, tags: {}'.format(X_test_content.shape, X_test_tags.shape))
del content_train, content_test, title_train, title_test, tags_train, tags_test
gc.collect()
# %%time
# del train, test
# X_train_sparse = csr_matrix(hstack([X_train_content, X_train_tags, X_train_feats.values])) 
# X_test_sparse = csr_matrix(hstack([X_test_content, X_test_tags, X_test_feats.values]))
# print(X_train_sparse.shape, X_test_sparse.shape)
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

X_train_sparse = load_sparse_csr(os.path.join(PATH, 'mediumeda/train_eda_csr.npz'))
X_test_sparse = load_sparse_csr(os.path.join(PATH, 'mediumedatest/test_eda_csr.npz'))
print(X_train_sparse.shape, X_test_sparse.shape)
%%time
def write_submission_file(prediction, path_to_sample=os.path.join(PATH, 'how-good-is-your-medium-article/sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv('submission.csv')
    
ridge = Ridge(random_state=17)                          
ridge_pred = ridge.fit(X_train_sparse, y_train).predict(X_test_sparse)      
write_submission_file(ridge_pred)
top30_plus = np.argsort(ridge.coef_)[-30:][::-1]
top30_plus
top30_minus = np.argsort(ridge.coef_)[:30]
top30_minus
feats_plus=[]
feats_minus=[]
for idx in top30_plus:
    if idx<X_train_content.shape[1]:
        feats_plus.append(list(cv_content.vocabulary_.keys())[list(cv_content.vocabulary_.values()).index(idx)])
#     elif (idx>=shape(X_train_content)[1] & idx<2*shape(X_train_content)[1]):
#         feats_plus.append(list(cv_title.vocabulary_.keys())[list(cv_title.vocabulary_.values()).index(idx)])
    elif idx>=(X_train_content.shape[1]+X_train_tags.shape[1]):
        feats_plus.append(X_train_feats.columns[idx-(X_train_content.shape[1]+X_train_tags.shape[1])])
    else:
        feats_plus.append(list(cv_tags.vocabulary_.keys())[list(cv_tags.vocabulary_.values()).index(idx-X_train_content.shape[1])])
for idx in top30_minus:
    if idx<X_train_content.shape[1]:
        feats_minus.append(list(cv_content.vocabulary_.keys())[list(cv_content.vocabulary_.values()).index(idx)])
#     elif (idx>=shape(X_train_content)[1] & idx<2*shape(X_train_content)[1]):
#         feats_minus.append(list(cv_title.vocabulary_.keys())[list(cv_title.vocabulary_.values()).index(idx)])
    elif idx>=(X_train_content.shape[1]+X_train_tags.shape[1]):
        feats_minus.append(X_train_feats.columns[idx-(X_train_content.shape[1]+X_train_tags.shape[1])])
    else:
        feats_minus.append(list(cv_tags.vocabulary_.keys())[list(cv_tags.vocabulary_.values()).index(idx-X_train_content.shape[1])])
plt.figure(figsize=(18,6))
ax1 = sns.barplot(feats_plus,ridge.coef_[top30_plus],color=color[2])
plt.ylabel('Value', fontsize=12)
plt.xlabel('Feature name', fontsize=12)
plt.title('Feature importance (positive coefficients)', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12)
    
plt.figure(figsize=(18,6))
ax2 = sns.barplot(feats_minus,ridge.coef_[top30_minus],color=color[3])
plt.ylabel('Value', fontsize=12)
plt.xlabel('Feature name', fontsize=12)
plt.title('Feature importance (negative coefficients)', fontsize=15)
plt.grid(False)
plt.xticks(rotation=90, fontsize=12);