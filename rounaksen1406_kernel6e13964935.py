import pandas as pd;

import numpy as np;

import scipy as sp;

import sklearn;

import sys;

from nltk.corpus import stopwords, wordnet

import nltk;

from nltk.stem import WordNetLemmatizer

from gensim.models import ldamodel

from gensim.models.hdpmodel import HdpModel

from gensim.models import CoherenceModel

from gensim import matutils, models

import gensim.corpora;

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer;

from sklearn.decomposition import NMF;

from sklearn.preprocessing import normalize;

import scipy.sparse

import string

import pickle;

import re;
stp = [i.lower() for i in stopwords.words('english')]

lemmatizer = WordNetLemmatizer()



mainData = pd.read_csv("../input/unstructured-l0-nlp-hackathon/data.csv")
def get_wordnet_pos(treebank_tag):

    '''

    Function that takes in nltk POS tags and returns tags so that they can be used for

    lemmatizaition

    '''

#     if treebank_tag.startswith('J'):

#         return wordnet.ADJ

#     elif treebank_tag.startswith('V'):

#         return wordnet.VERB

    if treebank_tag.startswith('N'):

        return wordnet.NOUN

#     elif treebank_tag.startswith('R'):

#         return wordnet.ADV

    else:

        return None
def get_lda_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,20)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in model.show_topics(10,20)}

    return pd.DataFrame.from_dict(word_dict)
def get_hdp_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,20)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in model.show_topics(10,20)}

    return pd.DataFrame.from_dict(word_dict)
def cleaner_array(A):

    tokens = nltk.word_tokenize(A)

    postags = nltk.pos_tag(tokens)



    resp_list = []



    for elem,tag in postags:

        if elem.lower().strip() not in stp and len(elem)>2:

            try:

                if get_wordnet_pos(tag) != None:

                    resp_list.append(lemmatizer.lemmatize(elem.lower().strip(), get_wordnet_pos(tag)))

            except:

                print(elem, tag)



    return(" ".join(resp_list))
mainData["clean"] = mainData["text"].apply(lambda x: cleaner_array(x))

mainData["clean"] = mainData["clean"].str.lower().apply(lambda x: re.sub(r'(@[\S]+)|(\w+:\/\/\S+)|(\d+)','',x))
mainData.columns
corpus = mainData["clean"].tolist()
countvect = CountVectorizer(stop_words='english', max_features = 4000, max_df = 0.8)

countvect_model = countvect.fit_transform(corpus)
data_count = pd.DataFrame(countvect_model.toarray(), columns=countvect.get_feature_names())

data_count.index = mainData.Id
corpus_count = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_count.transpose()))





id2word_count = dict((v, k) for k, v in countvect.vocabulary_.items())
lda_mod = models.LdaModel(corpus=corpus_count,

                          num_topics=5,

                          id2word=id2word_count,

                          random_state=1,

                          passes=50,#50,

                          alpha=0.001,

                          eta=1)
get_lda_topics(lda_mod,20)
corpus_transformed = lda_mod[corpus_count]



topics = []



for i in range(len(corpus_transformed)):

    v=dict(corpus_transformed[i])

    for top, score in v.items():

        if score == max(v.values()):

            topics.append(top)
final_output = pd.DataFrame(topics, index = mainData.Id, columns = ["topic"])



final_output.reset_index(inplace = True)
final_output.replace({'topic': {2: "glassdoor_reviews",

                                3: "Automobiles",

                                4: "sports_news",

                                1: "tech_news",

                                0: "room_rentals"}},

                     inplace = True)
final_output.to_csv("../input/output/submission_V9.csv", index=False)