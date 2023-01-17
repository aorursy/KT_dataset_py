import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from nltk.corpus import stopwords 

from wordcloud import WordCloud

%matplotlib inline

plt.style.use('fivethirtyeight')

plt.style.use('bmh')
business = pd.read_json('../input/yelp-dataset/yelp_academic_dataset_business.json',nrows=1000,lines=True)

business.isnull().any()
business['categories'].fillna(business['categories'].mode()[0], inplace=True)

business['attributes'].fillna(business['attributes'].mode()[0], inplace=True)

business['attributes'].fillna(method ='ffill', inplace=True)

business['hours'].fillna(business['hours'].mode()[0], inplace=True)

business['hours'].fillna(method ='ffill', inplace=True)

business
corpus = ' '.join(business['categories'])



corpus = pd.DataFrame(corpus.split(','),columns=['categories'])

cnt = corpus['categories'].value_counts().to_frame()[:20]

plt.figure(figsize=(14,8))

sns.barplot(cnt['categories'], cnt.index, palette = 'tab20')

plt.title('Top main categories listing');
plt.figure(figsize=(12,4))

ax = sns.countplot(business['stars'])

plt.title('Distribution of rating');
f,ax = plt.subplots(1,1, figsize=(14,8))

cnt = business['name'].value_counts()[:20].to_frame()



sns.barplot(cnt['name'], cnt.index, ax =ax)

ax.set_xlabel('')

ax.set_title('Top name of store in Yelp')



plt.subplots_adjust(wspace=0.3)

gc.collect();
tips = pd.read_json('../input/yelp-dataset/yelp_academic_dataset_tip.json',nrows=1000,lines=True)

tips=tips.dropna()

tips
cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(tips['text'].astype(str)))

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off');
tips['num_words'] = tips['text'].str.len()

tips['num_uniq_words'] = tips['text'].apply(lambda x: len(set(str(x).split())))

tips['num_chars'] = tips['text'].apply(lambda x: len(str(x)))

tips['num_stopwords'] = tips['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in set(stopwords.words('english'))]))

tips
f, ax = plt.subplots(2,2, figsize = (14,12))

ax1,ax2,ax3,ax4 = ax.flatten()

sns.distplot(tips['num_words'],bins=100,color='r', ax=ax1)

ax1.set_title('Distribution of Number of words')



sns.distplot(tips['num_uniq_words'],bins=100,color='b', ax=ax2)

ax2.set_title('Distribution of Unique words')



sns.distplot(tips['num_chars'],bins=100,color='y', ax=ax3)

ax3.set_title('Distribution of Char words')



sns.distplot(tips['num_stopwords'],bins=100,color='r', ax=ax4)

ax4.set_title('Distribution of Stop words');
tips['date'] = pd.to_datetime(tips['date'])

tips['year'] = tips['date'].dt.year

tips['month'] = tips['date'].dt.month



f,ax = plt.subplots(1,2,figsize = (14,6))

ax1,ax2 = ax.flatten()

cnt  = tips.groupby('year').sum()['compliment_count'].to_frame()

sns.barplot(cnt.index,cnt['compliment_count'], ax = ax1)

ax1.set_title('Distribution of stars by year')

ax1.set_ylabel('')



cnt  = tips.groupby('month').sum()['compliment_count'].to_frame()

sns.barplot(cnt.index,cnt['compliment_count'], ax = ax2)

ax2.set_title('Distribution of stars by month')

ax2.set_ylabel('');