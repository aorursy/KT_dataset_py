import pandas as pd

import numpy as np

import nltk

import pyLDAvis.gensim

from collections import Counter

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models import LdaMulticore,LdaModel

from gensim.corpora import Dictionary

from gensim.test.utils import common_corpus, common_dictionary, datapath, temporary_file

from multiprocessing import freeze_support

from gensim import corpora, models



def remove_nan(text):

    result = [i for i in text if not i == 'nan']

    return result

df=pd.read_csv("../input/reviews/reviews_clean.csv", index_col=0)
tokenizer = nltk.tokenize

df['Reviews_tokenized'] = df['Reviews'].apply(lambda x: tokenizer.word_tokenize(str(x)))

df['Reviews_tokenized'] = df['Reviews_tokenized'].apply(lambda x: remove_nan(x))

tokenized_reviews = df['Reviews_tokenized']



banks = df['Bank Name'].unique()



df2 = df.groupby('Bank Name')



review_dict = dict()



for bank in banks:

    df3 = df2.get_group(bank)

    review_list = df3['Reviews']

    bank_review = ' '.join(str(i) for i in review_list)

    review_dict[bank] = bank_review



dictionary = Dictionary(tokenized_reviews)



dictionary.filter_extremes(no_below = 25, no_above = 0.5, keep_n = 1000)



bow_corpus = [dictionary.doc2bow(reviews) for reviews in tokenized_reviews]



tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]



#lda_model = LdaModel(bow_corpus,num_topics=10,id2word=dictionary)



lda_model_tfidf = LdaModel(corpus_tfidf, num_topics=15, id2word=dictionary)

for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
for bank in banks:

    topics = tokenizer.word_tokenize(review_dict[bank])

    print('\n\nTopics for ',bank,' are:')

    bow_vector = dictionary.doc2bow(topics)

    for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):

        print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 7)))

data_ =  pyLDAvis.gensim.prepare(lda_model_tfidf, bow_corpus, dictionary=lda_model_tfidf.id2word)

pyLDAvis.display(data_)
