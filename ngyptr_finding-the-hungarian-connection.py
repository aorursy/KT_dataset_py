# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob 
from wordcloud import WordCloud,STOPWORDS
from xtractor import TopicExtractor as te

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import brown    

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Concat hungary related news
hungarian_data = pd.DataFrame()
datasets = ['reuters-newswire-2013.csv', 'reuters-newswire-2011.csv', 'reuters-newswire-2016.csv', 'reuters-newswire-2017.csv', 'reuters-newswire-2015.csv', 'reuters-newswire-2012.csv', 'reuters-newswire-2014.csv']
for ds in datasets:
    frame = pd.read_csv('../input/'+ds)
    hungary = frame[frame["headline_text"].str.lower().str.contains("hungary", na=False)]
    hungarian = frame[frame["headline_text"].str.lower().str.contains("hungarian", na=False)]
    hungarian_data = pd.concat([hungarian_data, hungary, hungarian])
    print(len(hungary), len(hungarian))
hungarian_data.reset_index(drop=True, inplace=True)
print('all: ' + str(len(hungarian_data)))
#Remove stopwords and lower text (will be great when it comes to feature extraction)
#Adding hungary to stopwords so it won't mess up wordclouds
STOPWORDS.add('hungary')
STOPWORDS.add('hungarian')
STOPWORDS.add('hungary\'s')
def preprocess(text):
    t = [x.lower() for x in text.split(' ') if x not in STOPWORDS]
    return ' '.join(t)
def explode(text):
    return [x.lower() for x in text.split(' ') if x not in STOPWORDS]

hungarian_data.headline_text = hungarian_data.headline_text.apply(preprocess)
hungarian_data = hungarian_data.assign(word_list=hungarian_data.headline_text.apply(explode))

hungarian_data.head()
#Extract publish year to be able to group articles
hungarian_data = hungarian_data.assign(year_publish=hungarian_data['publish_time'].apply(lambda x: str(x)[0:4]))

cnt_by_year =  pd.DataFrame(data = { 'years': [2011, 2012, 2013, 2014, 2015, 2016, 2017], 'cnt': [len(hungarian_data[hungarian_data.year_publish.astype('int64')==2011]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2012]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2013]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2014]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2015]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2016]),
                 len(hungarian_data[hungarian_data.year_publish.astype('int64')==2017])]})
#Number of articles on hungary in each year
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.set(xlabel='Year', ylabel='Number of articles')
sns.barplot(x=cnt_by_year.years, y=cnt_by_year.cnt, palette="Blues_d", ax=ax).set_title('Number of articles by year')
plt.show()
hungarian_data = hungarian_data.assign(word_count=hungarian_data['headline_text'].apply(lambda x: len(x.split(' '))))

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
sns.distplot(ax=ax,a=hungarian_data.word_count,axlabel="Word count").set_title('Distribution of title wordcounts')
wc_group = hungarian_data.groupby(['year_publish'])['word_count'].mean()

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.set(xlabel='Year published', ylabel='Average word count in heading')
sns.barplot(x=wc_group.keys(), y=wc_group.values, palette="Blues_d", ax=ax).set_title('Average word count in each year')
plt.show()
hungarian_data = hungarian_data.assign(month=hungarian_data['publish_time'].apply(lambda x: str(x)[4:6]))
to_plot = hungarian_data['month'].value_counts()
x = to_plot.index.tolist()
y = to_plot.values.tolist()
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
ax.set(xlabel='Month', ylabel='Number of articles')
sns.barplot(x=x, y=y, palette="Blues_d", ax=ax)

#How many times was Orbán (PM of Hungary) was mentioned in the observed article titles
orban_cnt = len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('orban')])
#How many times was corruption was mentioned in the observed article titles (surprisingly low given the corrupt nature of Fidesz (governing party since 2010))
corruption_cnt = len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('corrupt')]) + len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('corruption')])
#How many times was migrants was mentioned in the observed article titles (Migrants were and are all over the news in Hungary, fav. topic of fidesz)
migrant_cnt = len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('migrant')]) + len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('migration')])
#How many times was ngos was mentioned in the observed article titles (NGOs and civils are persecuted in the last 2 years)
ngo_cnt = len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('ngo')]) + len(hungarian_data[hungarian_data.headline_text.str.lower().str.contains('civil')])
print(orban_cnt, corruption_cnt, migrant_cnt, ngo_cnt)
#Lifted from https://www.kaggle.com/shivamb/seconds-from-disaster-text-analysis-fe-updated
def get_polarity(text):
    try:
        pol = TextBlob(text).sentiment.polarity
    except:
        pol = 0.0
    return pol

hungarian_data = hungarian_data.assign(sentiment=hungarian_data['headline_text'].apply(lambda x: get_polarity(x.lower())))

sort = hungarian_data.sort_values(by='sentiment')
pos = sort.tail(100)
neg = sort.head(100)

pos[['headline_text']].tail(10)
neg[['headline_text']].head(10)
positive = hungarian_data[hungarian_data['sentiment'] > 0.5]
negative = hungarian_data[hungarian_data['sentiment'] < 0.5]
neutral = hungarian_data[hungarian_data['sentiment'] == 0.5]
len(positive), len(negative), len(neutral)
def wordcloud_draw(data, color = 'white'):
    words = ' '.join(data)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(words)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
wordcloud_draw(positive['headline_text'])
wordcloud_draw(negative['headline_text'])
wordcloud_draw(neutral['headline_text'])
#Download and load the pre-trained embeddings on your local machine (not available on kaggle :()
#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
'''categories = [
            {
                "name": "economy",
                "keywords": ['money', 'bussiness', 'used', 'economy','credit', 'growth', 'entrepreneur', 'euro']
            }, 
            {
                "name": "sport",
                "keywords": ['ball','car' ,'rank','match', 'game', 'fan', 'stadium', 'sport', 'run', 'water']
            },
            {
                "name": "politics",
                "keywords": ['politics', 'pm', 'prime', 'minister', 'union', 'vote', 'voted', 'european', 'parliment', 'ngo']
            },
            {
                "name": "culture",
                "keywords": ['culture', 'music', 'exhibition', 'opening', 'paint', 'museum']
            },
            {
                "name": "crime",
                "keywords": ['kill', 'murder', 'blackmail', 'threat', 'death', 'chase']
            }
        ]

df = hungarian_data.word_list

extractor = te.TopicExtractor([model], categories)
results = extractor.extract(df)
'''
#results_frame = pd.DataFrame({"Results": results})
#hungarian_data = pd.concat([hungarian_data, results_frame], axis=1)
#hungarian_data.head()
#gr = hungarian_data.groupby(['Results']).count()
#gr