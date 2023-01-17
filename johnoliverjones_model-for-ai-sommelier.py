

import os

import pandas as pd

import numpy as np

from operator import itemgetter

from gensim import corpora, models, similarities

import gensim

import logging

from collections import OrderedDict

import tempfile

from nltk.corpus import stopwords

from string import punctuation

import pyLDAvis.gensim



#TEMP_FOLDER = tempfile.gettempdir()

#print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



df = pd.DataFrame()

df = pd.read_csv("../input/winemag-data-130k-v2.csv")

df.head(3)
# put all the wine descriptions into an numpy array called corpus

corpus=[]

a=[]

for i in range(len(df['description'])):

        a=df['description'][i]

        corpus.append(a)

corpus[0:2]
# remove common words and tokenize



stoplist = stopwords.words('english') + list(punctuation)



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]

dictionary = corpora.Dictionary(texts)



corpus = [dictionary.doc2bow(text) for text in texts]

#corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'wine.mm'), corpus)  # store to disk, for later use



tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors

total_topics = 30

# now using the vectorized corpus learn a LDA model

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi



#Show first 5 important words in the first 3 topics:

lda.show_topics(3,5)
data_lda = {i: OrderedDict(lda.show_topic(i,30)) for i in range(total_topics)}

#data_lda

df_lda = pd.DataFrame(data_lda)

print(df_lda.shape)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)

df_lda.head(5)
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel
row = 10007

print(df.loc[row,'variety'])

dt = lda[corpus[row]]

dt


dv = pd.DataFrame()

dv['Variety'] = df['variety']

for row in range(len(df)):

    dv.loc[row,'Likely_Topic'] = max(lda[corpus[row]], key = itemgetter(1))[0]

dv.head(5)
dv['Price'] = df['price']

dv['Title'] = df['title']

dv['Points'] = df['points']

dv['Vineyard'] = df['winery']

dv.head(15)
from textblob import TextBlob



dv['tb_Sentiment'] = 0

for row in range(len(df)):

    blob = TextBlob(df.loc[row,'description'])

    for sentence in blob.sentences:

        dv.loc[row,'tb_Sentiment'] += (sentence.sentiment.polarity)

dv.head(5)
description = ['brisk but sweet german wine. not too sweet']

texts = [word for word in str(description).lower().split() if word not in stoplist]

desc_v = dictionary.doc2bow(texts)

suggestion_types = lda[desc_v]

print(description)

suggestion_types
dv['Score'] = dv['tb_Sentiment']*10 + dv['Points']

t = pd.DataFrame()

t = dv.groupby(['Likely_Topic', 'Variety'])['Score'].agg([('Ave', 'mean'), ('Count', 'count')])

t = t.reset_index()

tt = t.loc[t['Count'].idxmax()] 

tt                                                                                                                                    