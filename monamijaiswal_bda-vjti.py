# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" di-rectory.

# For example, running this (by clicking run or pressing Shift+Enter)will list the files in the input directory



import os

print(os.listdir("../input"))







# Any results you write to the current directory are saved as output.

# data=pd.read_csv('../input/rediff-realtime-news-201710-201712/rediff_realtime_news_201710_201712_2.csv', sep='\t', engine='python')

data = pd.read_csv("../input/rediffwithcities/rediff_with_cities_2.csv")

cities = pd.read_csv('../input/worldcities/worldcities.csv', engine='python')

stop_words = pd.read_csv('../input/stop_words/stopwords.txt')

data.head()

# cities.head()

print(data.shape)

data = data.dropna();

data = data.drop_duplicates()

print(data.shape)
# import packages

import pandas as pd

pd.options.display.max_columns = 200

pd.options.mode.chained_assignment = None



from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

from string import punctuation



from collections import Counter

import re

import numpy as np



from tqdm import tqdm_notebook

tqdm_notebook().pandas()



from __future__ import print_function

import requests

from datetime import datetime

from tqdm import tqdm

from matplotlib import pyplot as plt
stop_words = []



f = open('../input/stop_words/stopwords.txt', 'r')

for l in f.readlines():

    stop_words.append(l.replace('\n', ''))

    

additional_stop_words = ['it', 'will']

stop_words += additional_stop_words



print(len(stop_words))
def getSources():

    source_url = 'https://newsapi.org/v1/sources?language=en'

    response = requests.get(source_url).json()

    sources = []

    for source in response['sources']:

        sources.append(source['id'])

    return sources



sources = getSources()

print('number of sources :', len(sources))

print('sources :', ', '.join(sources))



def mapping():

    d = {}

    response = requests.get('https://newsapi.org/v1/sources?language=en')

    response = response.json()

    for s in response['sources']:

        d[s['id']] = s['category']

    return d



m = mapping()

m['reuters'] = 'general'

m['money control'] = 'business'

m['news on air'] = 'general'

m['dna'] = 'general'

m['business standard'] = 'business'

m['med india'] = 'science'

m['zee news'] = 'general'

m['deccan herald'] = 'general'

m['open magazine'] = 'entertainment'

m['sify'] = 'general'

m['prokerala'] = 'health'

m['bollywood hungama'] = 'entertainment'

m['espn'] = 'sports'

m['asia net india'] = 'general'

m['india today'] = 'general'

m['huffington post'] = 'general'

m['state times'] = 'general'

m['money today'] = 'business'

m['current news india'] = 'general'

m['news.com.au'] = 'general'

m['the financial chronicle'] = 'business'

m['express uk'] = 'general'

m['trak.in'] = 'general'

m['science daily'] = 'science'

m['india prwire'] = 'general'

m['accommodation times'] = 'general'

m['equitymaster.com'] = 'business'

m['glamsham'] = 'entertainment'

m['linux for you'] = 'technology'

m['value research online'] = 'business'

m['construction world'] = 'business'

m['domain business'] = 'business'





print(list(set(m.values())))

sources = data.source.unique()

count = 0

for s in sources : 

    try:

        print(s ,':', m[s.lower()])

    except:

        print("Exception : ", s.lower())

        m[s.lower()] = 'others'

        count+=1

        continue

print(count)
data.source.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(16, 9))

# data.reset_index(inplace=True, drop=True)()

data.head()

data['category'] = ""

data['category'] = data['source'].map(lambda s: m[s.lower()])

data.to_csv("rediff_with_cities_2.csv", index=False)

data.shape
data.category.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(16, 9))

data.category.value_counts()
data = data.drop_duplicates('trimmed_description')

# data = data[~data['trimmed_description'].isnull()]



print(data.shape)

data = data[(data.trimmed_description.map(len) > 140) & (data.trimmed_description.map(len) <= 300)]

data.reset_index(inplace=True, drop=True)



print(data.shape)

data.trimmed_description.map(len).hist(figsize=(15, 5), bins=100)



# tokenizing each word for category distribution

# NO NEED TO RUN AGAIN, ALREADY DONE

from functools import reduce

def tokenizer(text):

    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]

    tokens = (reduce(lambda x,y: x+y, tokens))

    tokens = list([token for token in tokens if token not in (stop_words + list(punctuation))])

    return tokens





# data['trimmed_description'] = data['trimmed_description'].map(lambda d: d.decode('utf-8'))

data['tokens'] = data['trimmed_description'].progress_map(lambda d: tokenizer(d.lower()))

data.to_csv("rediff_with_tokens.csv")
for descripition, tokens in zip(data['trimmed_description'].head(5), data['tokens'].head(5)):

    print('description:', descripition)

    print('TOKENS', tokens)

    print() 

data.head()
def keywords(category):

    tokens = data[data['category'] == category]['tokens']

    alltokens = []

    for token_list in tokens:

        alltokens += token_list

    counter = Counter(alltokens)

    return counter.most_common(10)



# kw_tokens = []

for category in set(data['category']):

    print('category :', category)

#     kw_tokens[index] = keywords(category)

    print('top 10 keywords:', keywords(category))

    print('---')

    

# for x in kw_tokens:

#     print(x)

    
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 2), stop_words='english')

vz = vectorizer.fit_transform(list(data['tokens'].map(lambda tokens: ' '.join(tokens))))



vz.shape



tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')

tfidf.columns = ['tfidf']



tfidf.tfidf.hist(bins=25, figsize=(15,7))

from wordcloud import WordCloud



def plot_word_cloud(terms):

    text = terms.index

    text = ' '.join(list(text))

    # lower max_font_size

    wordcloud = WordCloud(max_font_size=40).generate(text)

    plt.figure(figsize=(25, 25))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("on")

    plt.show()

plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))
