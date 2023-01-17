# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ca_data = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')
ca_data.head()
channel_dataset = pd.read_csv('../input/youtube-channels-100000/channels.csv')
channel_dataset.head()
channel_dataset.info()

with open('/kaggle/input/youtube-new/CA_category_id.json') as f:
    categories = json.load(f)
categories_map = {}
for category in categories['items']:
    categories_map[int(category['id'])] = category['snippet']['title']
ca_data['category_id'] = ca_data['category_id'].map(categories_map)
ca_data.head()
print('info', ca_data.info())
print('Views stats', ca_data['views'].describe())
print('Comment stats', ca_data['comment_count'].describe())
print('Like stats', ca_data['likes'].describe())
ca_data.isnull().sum()
channel_dataset = channel_dataset.rename(columns={'title':'channel_title', 'category_name': 'channel_category_name'})
new_ca_data= ca_data.merge(channel_dataset,how='left', on='channel_title')

print('missing values percentage', new_ca_data['followers'].isnull().sum()/len(new_ca_data))
new_ca_data.info()
ca_data['description'] = ca_data['description'].fillna(value='None',inplace=True)
ca_data['publish_time'] = pd.to_datetime(ca_data['publish_time'], format = '%Y-%m-%dT%H:%M:%S.%fZ')
ca_data['publish_month'] = ca_data['publish_time'].dt.month
ca_data['publish_hour'] = ca_data['publish_time'].dt.hour
ca_data['like_percentage'] = (ca_data['likes'] / ca_data['views']) * 100
ca_data['dislike_percentage'] = (ca_data['dislikes']/ca_data['views']) * 100
ca_data.loc[ca_data['likes'] >= ca_data['dislikes'],'more_likes'] = 1
ca_data.loc[ca_data['likes'] < ca_data['dislikes'],'more_likes'] = 0

def get_title_length(title):
    return len(title)
def get_number_of_tags(string_of_tags):
    tags = string_of_tags.split('|')
    tags_no_empty_strings = []
    for tag in tags:
        if tag != '':
            tags_no_empty_strings.append(tag)
    return len(tags_no_empty_strings)

ca_data['title_length'] = ca_data.apply(lambda row: get_title_length(row['title']), axis=1)
ca_data['number_of_tags'] = ca_data.apply(lambda row: get_number_of_tags(row['tags']), axis=1)
def contains_all_caps_word(title):
    words = title.split(' ')
    for word in words:
        if word.isupper():
            return True 
print(contains_all_caps_word('HEY how are you'))
    
f,ax = plt.subplots(3,1, figsize=(20,10))
views_by_categories_plot = sns.barplot(x=ca_data['category_id'], y=ca_data['views'], ax=ax[0])
views_by_categories_plot.set_title('Number of Views Per Category') 
likes_by_categories_plot = sns.barplot(x=ca_data['category_id'], y=ca_data['likes'], ax=ax[1])
likes_by_categories_plot.set_title('Number of Likes Per Category')     
comment_count_by_categories_plot = sns.barplot(x=ca_data['category_id'], y=ca_data['comment_count'], ax=ax[2])
comment_count_by_categories_plot.set_title('Number of Comments Per Category')  
plt.tight_layout()
plt.figure(figsize=(16,6))
correlation_matrix_heatmap = sns.heatmap(ca_data[['views', 'likes', 'comment_count','dislikes', 'title_length', 'number_of_tags']].corr(),vmin=-1,vmax=1, cmap='YlGnBu', annot=True)
correlation_matrix_heatmap.set_title('Correlation Heatmap')
f,ax = plt.subplots(2,1, figsize=(16,6))
views_by_hour = sns.lineplot(x=ca_data['publish_hour'], y=ca_data['views'],data=ca_data,marker='o', ax=ax[0])
views_by_hour.set_title('Views per Publish Hour')
views_by_month= sns.lineplot(x=ca_data['publish_month'], y=ca_data['views'],data=ca_data,marker='o', ax=ax[1])
views_by_month.set_title('Views per Publish Month')
plt.tight_layout()
plt.figure(figsize=(16,6))
feeling_barplot = sns.barplot(x=ca_data['category_id'], y=ca_data['views'], hue=ca_data['more_likes'], data=ca_data)
feeling_barplot.set_xticklabels(feeling_barplot.get_xticklabels(),rotation=30)
feeling_barplot.set_title('Views Per Category Per Feeling')
plt.tight_layout()
ca_data = ca_data.drop(['comment_count'], axis=1)
def get_title_length(title):
    return len(title)
def get_number_of_tags(string_of_tags):
    tags = string_of_tags.split('|')
    tags_no_empty_strings = []
    for tag in tags:
        if tag != '':
            tags_no_empty_strings.append(tag)
    return len(tags_no_empty_strings)
ca_data = ca_data.drop(['video_id', 'channel_title', 'description', 'likes', 'dislikes', 'thumbnail_link', 'comment_count', 'video_error_or_removed'],axis=1)
#Convert boolean valus to integers
ca_data = ca_data.astype({'comments_disabled': 'int', 'ratings_disabled': 'int'})
ca_data['publish_time'] = ca_data.apply(lambda row: get_publish_hour(row['publish_time']), axis=1)
ca_data = ca_data.rename(columns={'publish_time':'publish_hour'})
ca_data = pd.get_dummies(ca_data,prefix='category', columns=['category_id'])
ca_data = pd.get_dummies(ca_data,prefix='hour', columns=['publish_hour'])

ca_data['title_length'] = ca_data.apply(lambda row: get_title_length(row['title']), axis=1)
ca_data = ca_data.drop(columns=['title'])
ca_data['number_of_tags'] = ca_data.apply(lambda row: get_number_of_tags(row['tags']), axis=1)
ca_data = ca_data.drop(['tags', 'trending_date'],axis=1)

y = ca_data['views']
X = ca_data[ca_data.columns.difference(['views'])]
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))
