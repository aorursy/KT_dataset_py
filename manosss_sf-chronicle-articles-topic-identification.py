#import libraries

import pandas as pd

import re

from string import punctuation

import nltk

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from gensim import corpora, models

import gensim

from gensim.models import CoherenceModel

import pyLDAvis

import pyLDAvis.gensim 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/data.csv')

data['recent']=False

for i in range(len(data['date_updated'])):

    if '2019' in data['date_updated'][i]: # If the update was made in 2019, mark article as recent

        data['recent'][i]=True

data
recent=data.loc[(data['recent']==True),['text','title']]
def strip_punctuation(s):

    mypunclist=list(punctuation)+['’','“','”','،','—','‘']

    return ''.join(c for c in s if c not in mypunclist)



def tokenizer(text):    

    tokenized=word_tokenize(text)

    return tokenized



def lemmatizer(mylist):

    lem = WordNetLemmatizer()

    lemmatized = [lem.lemmatize(i,pos="v") for i in mylist]

    return(lemmatized)
recent['text']=recent['text'].str.lower()

recent['text'] = recent['text'].apply(strip_punctuation)

stop = stopwords.words('english')

recent['text'] = recent['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

recent['text'] = recent['text'].apply(tokenizer)

recent ['tokenized']=recent['text'].apply(lemmatizer)
l=list(recent['tokenized'].values)
flat_list = [item for sublist in l for item in sublist]

count=pd.Series(flat_list).value_counts()[:20]

count.to_frame('Count')
a= ['say','would','could','one','also','san','us','go','two','first','also','make','find','take','many','like','new','get','use','back','year','time','day','dont','might','part','still','come']

for i in l:

    for j in a:

        while j in i : i.remove(j)
flat_list = [item for sublist in l for item in sublist]

count=pd.Series(flat_list).value_counts()[:20]

count.to_frame('Count')
dictionary = corpora.Dictionary(l)

corpus = [dictionary.doc2bow(text) for text in l]

ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=3,passes=20,id2word = dictionary,random_state=0)
print(ldamodel.print_topics(num_topics=3, num_words=4))
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

vis
data
business_data=data.loc[(data['category']=='Biz+Tech'),['text']]
business_data['text']=business_data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

business_data['text'] = business_data['text'].apply(strip_punctuation)

stop = stopwords.words('english')

business_data['text'] = business_data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

business_data['text'] = business_data['text'].apply(tokenizer)

business_data['tokenized']=business_data['text'].apply(lemmatizer)
l=list(business_data['tokenized'].values)

flat_list = [item for sublist in l for item in sublist]

count=pd.Series(flat_list).value_counts()[:20]

count.to_frame('Count')
a= ['say','also','pge','us','go','get']

for i in l:

    for j in a:

        while j in i : i.remove(j)
dictionary = corpora.Dictionary(l)

corpus = [dictionary.doc2bow(text) for text in l]

ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=2,id2word = dictionary,random_state=0)
print(ldamodel.print_topics(num_topics=2, num_words=5))

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

vis
us_data=data.loc[(data['category']=='US & World'),['text']]

us_data['text']=us_data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

us_data['text'] = us_data['text'].apply(strip_punctuation)

stop = stopwords.words('english')

us_data['text'] = us_data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

us_data['text'] = us_data['text'].apply(tokenizer)

us_data['tokenized']=us_data['text'].apply(lemmatizer)
l=list(us_data['tokenized'].values)

flat_list = [item for sublist in l for item in sublist]

count=pd.Series(flat_list).value_counts()[:20]

count.to_frame('Count')
a= ['say','would','also','us','go']

for i in l:

    for j in a:

        while j in i : i.remove(j)

dictionary = corpora.Dictionary(l)

corpus = [dictionary.doc2bow(text) for text in l]

ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=2,id2word = dictionary,random_state=0)
print(ldamodel.print_topics(num_topics=2, num_words=5))

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

vis
food_data=data.loc[(data['category']=='Food'),['text']]

food_data['text']=food_data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

food_data['text'] = food_data['text'].apply(strip_punctuation)

stop = stopwords.words('english')

food_data['text'] = food_data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

food_data['text'] = food_data['text'].apply(tokenizer)

food_data['tokenized']=food_data['text'].apply(lemmatizer)
l=list(food_data['tokenized'].values)

flat_list = [item for sublist in l for item in sublist]

count=pd.Series(flat_list).value_counts()[:20]

count.to_frame('Count')
a= ['say','san','one','get','also','would']

for i in l:

    for j in a:

        while j in i : i.remove(j)

dictionary = corpora.Dictionary(l)

corpus = [dictionary.doc2bow(text) for text in l]

ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics=2,id2word = dictionary,random_state=0)
print(ldamodel.print_topics(num_topics=2, num_words=5))

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

vis