import pandas as pd

import nltk
data = pd.read_excel('../input/sentimnt-data/Voda Pos.xlsx', error_bad_lines=False)
data['tweet_id'] = data['tweet_id'].astype(str)

data['user_id'] = data['user_id'].astype(str)
data_text = data[['text']]

data_text['index'] = data_text.index

documents = data_text
print(len(documents))

print(documents[:5])
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(2018)

import nltk

nltk.download('wordnet')

NEWSTOPWORDS = ['#vodafone', '#vodafonein', '@idea', '#ideacellular', '@vodafonein', '@vodafoneideabiz', '@vodafonegroup', '@vodafonebiz', '@idea_cares', '@vodaidea_news', '@dot_india', '#vodafoneidea', '#vodafoneindia', '#vodaphoneinida', '#ideaindia', '@vodafonegroup', '@adityabirlagrp', '@trai', 'bsnl', 'idea', 'vodafone', '@bsnlcorporate', '#bsnl', '@bsnl_gj', 'bsnl', '@cmdbsnl', '#bhartiairtel','#airtel', '#airtelthanksapp', '@airtelindia', 'bhartiairtel','airtel', 'airtelthanksapp', 'airtelindia', '@airtel_presence', 'airtel_presence', '#airtel_presence', '@airtelnews', 'airtelpresence', 'airtelnews', '#telecom', '@jiocare', '#jio', '#reliancejio', '#jio4g', 'jio', '@reliancejio', 'reliancejio', 'reliance', '@Paytmcare', '@Paytm', '@PaytmBank', '@PaytmMoneyCare', '@PaytmMoney', 'twitter', 'pic', 'even', 'will', 'now', 'vodafonein', 'time', 'day', 'people', 'guy', 'one', 'day', '#guy', 'jiocare', '#idea_cares', '#idea_care', 'idea_care', '#vodafoneidea', 'vodafoneidea', 'idea_cares', 'india', 'going', 'company', 'getting', 'https', 'recipient_id', 'messages', 'id', 'dear', 'please', 'using', 'hi', 'really', 'vodafonein', 'airtelindia', 'raha', 'karwa', 'walo', 'apko', 'bilkul', 'googl', 'kaam', 'want', 'give', 'pictwittercom', 'se', 'best', 'twittercom', 'hai', '2019', 'concern', 'prices', '91 9702140710', 'see', 'sir', 'birthday', 'ha', 'giving', 'कर', 'नह', 'don', 'strongertogether', 'kindly', 'tone', 'ke', 'ho', 'nahi', 'ka', 'ur', 'us']

#STOPWORDS.extend(NEWSTOPWORDS)

STOPWORDS = STOPWORDS.union(NEWSTOPWORDS)
def lemmatize_stemming(text):

    stemmer = SnowballStemmer('english')

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result
processed_docs = documents['text'].map(preprocess)

processed_docs[:10]

print(len(processed_docs))
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 10:

        break
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

bow_corpus[3778]
#bagofwords

bow_doc_3778 = bow_corpus[3778]

for i in range(len(bow_doc_3778)):

    print("Word {} (\"{}\") appears {} time.".format(bow_doc_3778[i][0], 

                                               dictionary[bow_doc_3778[i][0]], 

bow_doc_3778[i][1]))
#tf-idf

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:

    pprint(doc)

    break
#ldausingbow

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
#ldausingtfidf

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx, topic))
processed_docs[3778]
#classifyusingldabow

for index, score in sorted(lda_model[bow_corpus[3778]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
#classifyingusingtfidf

for index, score in sorted(lda_model_tfidf[bow_corpus[3778]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))