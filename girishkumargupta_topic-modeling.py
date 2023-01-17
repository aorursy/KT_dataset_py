import pandas as pd

import numpy as np

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt

import re

import spacy
emails = pd.read_csv("../input/enron-email-dataset/emails.csv")

emails.head()
emails.shape
email_subset = emails[:10000]

print(email_subset.shape)

print(email_subset.head())
def parse_into_emails(messages):

    emails = [parse_raw_message(message) for message in messages]

    return {

        'body': map_to_list(emails, 'body'),

        'to': map_to_list(emails, 'to'),

        'from_': map_to_list(emails, 'from')

    }
# cleaning 

def parse_raw_message(raw_message):

    lines = raw_message.split('\n')

    email = {}

    message = ''

    keys_to_extract = ['from', 'to']

    for line in lines:

        if ':' not in line:

            message += line.strip()

            email['body'] = message

        else:

            pairs = line.split(':')

            key = pairs[0].lower()

            val = pairs[1].strip()

            if key in keys_to_extract:

                email[key] = val

    return email
def map_to_list(emails, key):

    results = []

    for email in emails:

        if key not in email:

            results.append('')

        else:

            results.append(email[key])

    return results
email_df = pd.DataFrame(parse_into_emails(emails.message))

email_df = email_df.loc[email_df['to'] == email_df['from_']]

email_df.to_csv('from_to_same.csv',index=False)





print(email_df.head())
email_df.shape

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
print(email_df.iloc[1]['body'])
data = email_df.body.values.tolist()

type(data)
def sent_to_words(sentences):

    for sentence in sentences:

        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))
print(data_words[3])
from gensim.models.phrases import Phrases, Phraser
# Build the bigram and trigram models

bigram = Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

trigram = Phrases(bigram[data_words], threshold=100)
bigram_mod = Phraser(bigram)

trigram_mod = Phraser(trigram)
print(trigram_mod[bigram_mod[data_words[200]]])
def remove_stopwords(texts):

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def make_bigrams(texts):

    return [bigram_mod[doc] for doc in texts]



def make_trigrams(texts):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_out = []

    for sent in texts:

        doc = nlp(" ".join(sent))

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[200])
# create dictionary and corpus both are needed for (LDA) topic modeling



# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
# Build LDA model

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=20,

                                           random_state=100,

                                           update_every=1,

                                           chunksize=100,

                                           passes=10,

                                           alpha='auto',

                                           per_word_topics=True)
print(lda_model.print_topics())