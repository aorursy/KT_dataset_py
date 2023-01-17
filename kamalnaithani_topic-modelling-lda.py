# import libraries  

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk import pos_tag_sents, word_tokenize

from nltk.tokenize import wordpunct_tokenize

from pprint import pprint

import re, random, os

import string



# spacy for basic preprocessing, optional, can use nltk as well (lemmatisation etc.)

import spacy



# gensim for LDA 

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# stop words

# string.punctuation (from the 'string' module) contains a list of punctuations

from nltk.corpus import stopwords

stop_words = stopwords.words('english') + list(string.punctuation)
df = pd.read_csv('../input/amazondataset/7817_1.csv')
df.head()
# filter for product id = amazon echo

df = df[df['asins']=="B01BH83OOM"]

df.head()
# tokenize words and clean 

def sent_to_words(sentences, deacc=True): # deacc=True removes punctuations

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence)))  





# convert to list

data = df['reviews.text'].values.tolist()

data_words = list(sent_to_words(data))



print(data_words[3])
# Define functions for stopwords and lemmatization

def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
# call functions



# remove Stop Words

data_words_nostops = remove_stopwords(data_words)



# initialize spacy 'en' model, keeping only tagger component (for efficiency)

# python3 -m spacy download en

nlp = spacy.load('en', disable=['parser', 'ner'])



# Do lemmatization keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])



print(data_lemmatized[3])
# compare the nostop, lemmatised version with the original one

print(data_words[3])
# create dictionary and corpus

# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]



# View

print(corpus[2])
# Human readable format of corpus (term-frequency)

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=10, 

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
# Compute Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.



# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis