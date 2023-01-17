import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import numpy as np

import json
ut = pd.read_csv('../input/youtube-new/INvideos.csv')

ut.head()
id_dict = {}

with open('../input/youtube-new/IN_category_id.json') as f:

    data = json.load(f)

    for category in data['items']:

        id_dict[int(category['id'])] = category['snippet']['title']

        

ut['category_id'] = ut['category_id'].map(id_dict)
ut.head(2)
ut.info()
ut['publish_time'] = pd.to_datetime(ut['publish_time'])

ut['trending_date'] = pd.to_datetime(ut['trending_date'],format="%y.%d.%m")

ut['year'] = ut['publish_time'].dt.year
ut['date_diff'] = (ut['trending_date'].dt.date - ut['publish_time'].dt.date).dt.days
ut.info()
ut.sample(3)
plt.figure(figsize=(8,4))

sns.heatmap(ut.isnull(),cmap = 'winter',cbar=False,yticklabels=False)
ut.isnull().sum()
ut.dropna(axis=0,inplace=True)
ut.isnull().sum()
ut.drop(labels=['tags','thumbnail_link','comments_disabled',

                'ratings_disabled','video_error_or_removed','description'],axis=1,inplace=True)
ut.sample(2)
ut.shape
ut.title.nunique()
ut.title.value_counts()
ut.drop_duplicates(subset='title',keep='first',inplace=True)
ut.shape
ut.describe()
co = ['views','likes','dislikes','comment_count']

plt.figure(figsize=(20,10))

for i in range(0,len(co)):

    plt.subplot(5,1,i+1)

    sns.boxplot(ut[co[i]],color='green',fliersize=9,orient='h')

    plt.tight_layout()
plt.figure(figsize=(10,12))

for i in range(0,len(co)):

    plt.subplot(5,1,i+1)

    sns.distplot(ut[co[i]],kde=True)

    plt.tight_layout()
plt.figure(figsize=(12,6))

sns.heatmap(ut[['views','likes','dislikes','comment_count']].corr(),annot=True,cmap= 'Blues')
plt.figure(figsize=(10,4))

sns.countplot(x='category_id',data=ut,order=ut['category_id'].value_counts().index,palette='Blues_r')

plt.xlabel('Category')

plt.xticks(rotation=90)

plt.title('Categorys with the highest views')
ut.date_diff.value_counts().head(10).plot(kind='bar')

plt.xlabel('Number of days')

plt.ylabel('Count')

plt.title('Higest trend within the days')
z_days = ut.query('date_diff==0')
z_days['category_id'].value_counts().plot(kind='bar')
ent = z_days.query('category_id=="Entertainment"')

ent
sns.countplot(x='year',data=ent)
from PIL import Image

from wordcloud import WordCloud,ImageColorGenerator
clud = WordCloud(min_font_size=4,background_color='white',max_words=1200).generate(str(ent['title'])) 

plt.figure(figsize=(10,18))

plt.imshow(clud)

plt.axis('off')
edu = z_days.query('category_id=="Education"')
sns.countplot(x='year',data=edu)
wrd = WordCloud(min_font_size=4,background_color='white',max_words=1200).generate(str(edu['title'])) 

plt.figure(figsize=(10,18))

plt.imshow(wrd)

plt.axis('off')