# if set to False, the notebook takes about 10 minutes to run

load_preprocessed_file = True
final_df_filename = 'df_final_covid_clean_topics.pkl'



import numpy as np

import pandas as pd

import os

import glob

import json



import pickle as pkl

import string

import nltk

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

from nltk.tokenize import sent_tokenize#, word_tokenize

from nltk.corpus import stopwords

import time

from multiprocessing import Pool

import numpy as np

import multiprocessing

from collections import Counter

from itertools import chain

import operator

from gensim.models.phrases import Phrases, Phraser

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import itertools

import collections

import random

from gensim.corpora import Dictionary

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



import scipy.sparse as ss



# https://github.com/gregversteeg/corex_topic

!pip install 'corextopic'



from corextopic import corextopic as ct

from corextopic import vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice

import matplotlib.pyplot as plt



%matplotlib inline



from ipywidgets import interact

import ipywidgets as widgets
if load_preprocessed_file is False:



    # credit: https://www.kaggle.com/gkaraman/topic-modeling-lda-on-cord-19-paper-abstracts

    df = pd.read_csv(

        '/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv'

        , dtype={

            'Microsoft Academic Paper ID': str

            ,'pubmed_id': str

        })



    # Some papers are duplicated since they were collected from separate sources. Thanks Joerg Rings

    duplicate_paper = ~(df.title.isnull() | df.abstract.isnull()) & (df.duplicated(subset=['title', 'abstract']))

    df = df[~duplicate_paper].reset_index(drop=True)



    df = df.dropna(subset=['abstract'])



    # create a column that appends title+abstract. This column will be the "document" that all searching/clustering/vectorization will use

    df['document'] = df['title'] + '. ' + df['abstract']



    print(df.shape)
# "supercalifragili-\nsticexpialidocious\nthis is a new line" -> "supercalifragilisticexpialidocious this is a new line"

def clean_newlines(text):

    text = text.replace('-\n', '')

    text = text.replace('\n', ' ').replace('\r',' ')

    

    return text



test = 'supercalifragili-\nsticexpialidocious\nthis is a new line '

clean_newlines(test)
def clean_chars(text):

    text = "".join(i for i in text if ord(i) < 128) # remove all non-ascii characters

    text = text.replace('\t', ' ') # convert a tab to space

    # fastest way to remove all punctuation (except ' . and !) and digits

    text = text.replace('[Image: see text]', '')

    text = text.translate(str.maketrans('', '', '"#$%&()*+,-/:;<=>@[\\]^_`{|}~' + string.digits))

    

    return text.strip()



clean_chars('[Image: see text] Numbers 123 are greater than 456?!\t"I\'m of the op1ni0n it isn\'t..."')
# credit: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

# helper correctly accounts for the same word having a different

# part-of-speech depending on the context of its usage

def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



sentence = "The ten foot striped bats are hanging on their good better best feet. The bat's wings were ten feet wide."



print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
def dedupe_sentences(sentences):

    deduped = []

    for s in sentences:

        if s not in deduped:

            deduped.append(s)

    

    return deduped



test_sentences = [

    ['see', 'figure', 'for', 'data'],

    ['not', 'cleared', 'for', 'release'],

    ['new', 'sentence', 'here'],

    ['see', 'figure', 'for', 'data'],

    ['not', 'cleared', 'for', 'release'],

    ['finally']

]



dedupe_sentences(test_sentences)
stpwrds_list = stopwords.words('english')



# add custom stopwords here, discovered from most common words and ngrams (further below)

stpwrds_list += ['...', 'also', 'could', 'thus', 'therefore']



stpwrds_lower = [wrd.lower() for wrd in stpwrds_list]



stpwrds_list += stpwrds_lower



stpwrds = set(stpwrds_list) # dedupes stopwords
# given a string representing an entire document, returns the following format where all the words are non-stopwords and lemmatized:

#example = [

#    ['Sentence', 'one', 'words'],

#    ['Sentence', 'two', 'words']

#]



def clean(text, min_word_len=3, lower=True):

    

    if (lower is True):

        text = text.lower()

    

    text = clean_newlines(text)

    text = clean_chars(text)

    

    sentences = sent_tokenize(text)

    

    clean_sentences = []

    

    for s in sentences:

        clean_sent_words = [

            lemmatizer.lemmatize(w, get_wordnet_pos(w))

            for w in nltk.word_tokenize(s)

            # skip short words, contraction parts of speech, and storwords

            if len(w) >= min_word_len and w[0] != '\'' and w not in stpwrds

        ]

        

        clean_sentences.append(clean_sent_words)

    

    # one and only one identical sentence is allowed per document

    # this helps avoid common phrases like "Table of data below:" appearing

    # many times, which will skew the word associations

    clean_sentences = dedupe_sentences(clean_sentences)

    

    return clean_sentences



test = " NOT CLEARED FOR PUBLIC RELEASE. See table for references. The ss goes here. "

test += "US interests. U.S. Enterprise. The ten foot striped bats are hanging on their good better best feet. The "

test += "bat's Wings were ten feet wide. Is the U.S. Enterprise 123 better than 456?!\t\"I\'m of the op1ni0n it isn\'t...\""

test += " New sentence. NOT CLEARED FOR PUBLIC RELEASE. See table for references."

clean(test)
%%time



# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

def parallelize_dataframe(df, func, n_cores=multiprocessing.cpu_count()):

    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()



    return df



def clean_dataframe(df):

    df['clean'] = df.apply(lambda x: clean(x['document']), axis=1)



    return df



if load_preprocessed_file is False:

    # this parallelize_dataframe way takes about 7 minutes

    df = parallelize_dataframe(df, clean_dataframe)



    # the "swifter" keyword/library aims to make dataframe processing faster, but it didn't help in this case

    # !pip install 'swifter'

    # import swifter

    # https://towardsdatascience.com/add-this-single-word-to-make-your-pandas-apply-faster-90ee2fffe9e8

    # this method was slower at 17 minutes in total, but it provided a nice progress bar and countdown timer

    # df['clean'] = df.swifter.apply(lambda x: clean(x['document']), axis=1)



    df[['clean']].head(3)

else:

    df = pkl.load(open('/kaggle/input/cached-data-interactive-abstract-and-expert-finder/' + final_df_filename, "rb" ))
clean_words = df['clean'].tolist()

clean_words = [item for sublist in clean_words for item in sublist]



print(str(len(clean_words)) + ' total words in corpus')



counter_obj = Counter(chain.from_iterable(clean_words))

word_counts = counter_obj.most_common()

word_counts.sort(key=operator.itemgetter(1), reverse=True)



word_counts[0:10]
%%time

# higher threshold means fewer ngrams - open question, how to optimize these hyperparams?



bigram = Phrases(clean_words, min_count=384, threshold=64, delimiter=b'_')

trigram = Phrases(bigram[clean_words], min_count=64, threshold=32, delimiter=b'_')
sorted(

    {k:v for k,v in bigram.vocab.items() if b'_' in k if v>=bigram.min_count and str(k).count('_') == 1}.items(),

    key=operator.itemgetter(1),

    reverse=True

)
sorted(

    {k:v for k,v in trigram.vocab.items() if b'_' in k if v>=trigram.min_count and str(k).count('_') == 2 }.items(),

    key=operator.itemgetter(1),

    reverse=True

)
def get_ngram_words(sent_arr):

    result = []



    for s in sent_arr:

        sent_result = []

        for w in trigram[bigram[s]]:

            if (w not in stpwrds): # we need to check again, because we may have added ngrams to the stopword list

                sent_result.append(w)



        result.append(sent_result)

    return result



test_sentences = [

    ['polymerase', 'chain', 'reaction'],

    ['infectious', 'disease', 'cause', 'unknown', 'public', 'health'],

    ['significant', 'acute', 'respiratory', 'disease', 'report', 'world', 'health']

]



get_ngram_words(test_sentences)
%%time



def convert_ngram_dataframe(df):

    df['clean'] = df.apply(lambda x: get_ngram_words(x['clean']), axis=1)



    return df



if load_preprocessed_file is False:

    df = parallelize_dataframe(df, convert_ngram_dataframe)



    df[['clean']].head(3)
all_words = []

docs = []



for index, row in df.iterrows():

    sent_arr = row['clean']

    doc_words = []

    

    for s in sent_arr:

        for w in s:

            doc_words.append(w)

            all_words.append(w)

    

    docs.append(doc_words)



print('TOTAL WORDS: ' + str(len(all_words)))

print('UNIQUE WORDS: ' + str(len(set(all_words))))
%%time



# credit: https://www.kaggle.com/gkaraman/topic-modeling-lda-on-cord-19-paper-abstracts



# Create a dictionary representation of the documents

dictionary = Dictionary(docs)

dictionary.filter_extremes(no_below=32, no_above=0.2)



# Create Bag-of-words representation of the documents

#corpus = [dictionary.doc2bow(doc) for doc in docs]



print('Number of unique tokens: %d' % len(dictionary))

#print('Number of documents: %d' % len(corpus))





def remove_non_dict_words(sent_arr):

    result = []



    for s in sent_arr:

        for w in s:

            if w in dictionary.token2id:

                result.append(w)

                

    return result



def remove_non_dict_words_df(df):

    df['clean_tfidf'] = df.apply(lambda x: remove_non_dict_words(x['clean']), axis=1)



    return df



df = parallelize_dataframe(df, remove_non_dict_words_df)



df = df.reset_index() # after all the processing, there are some gaps in the indices, so we reset them to make index counting easier later
%%time



if load_preprocessed_file is False:



    def dummy(doc):

        return doc



    vectorizer = CountVectorizer(

        tokenizer=dummy,

        preprocessor=dummy,

    )  



    corex_docs = df['clean_tfidf'].tolist()

    doc_word = vectorizer.fit_transform(corex_docs)



    doc_word = ss.csr_matrix(doc_word)



    # Get words that label the columns (needed to extract readable topics and make anchoring easier)

    words = list(np.asarray(vectorizer.get_feature_names()))



    #doc_word.shape # n_docs x m_words





    # https://github.com/gregversteeg/corex_topic

    # Train the CorEx topic model with x topics (n_hidden)

    topic_model = ct.Corex(n_hidden=12, words=words, max_iter=500, verbose=False, seed=2020)

    #topic_model.fit(doc_word, words=words)



    topic_model.fit(doc_word, words=words)





    plt.figure(figsize=(10,5))

    plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)

    plt.xlabel('Topic', fontsize=16)

    plt.ylabel('Total Correlation (nats)', fontsize=16);

    # no single topic should contribute too much. If one does, that indicates more investigation for boilerplate text, more preprocessing required

    # To find optimal num of topics, we should keep adding topics until additional topics do not significantly contribute to the overall TC

    

    pkl.dump(topic_model, open('corex_topic_model.pkl', "wb"))

else:

    topic_model = pkl.load(open('/kaggle/input/cached-data-interactive-abstract-and-expert-finder/corex_topic_model.pkl', "rb" ))



# Print all topics from the CorEx topic model

topics = topic_model.get_topics()

topic_list = []



for n,topic in enumerate(topics):

    topic_words,_ = zip(*topic)

    print('{}: '.format(n) + ','.join(topic_words))

    topic_list.append('topic_' + str(n) + ': ' + ', '.join(topic_words))
%%time



if load_preprocessed_file is False:



    # remove any existing topic columns. This allows us to iterate on number of topics

    for c in [col for col in df.columns if col.startswith('topic_')]:

        del df[c]



    # TODO: inefficient code. Ideas to improve this: for each topic, first create a np array of length of rows, then iterate

    # over those indices setting the scores with the rest default to 0, then set the whole df col

    for topic_num in range(0, len(topic_model.get_topics())):

        df['topic_' + str(topic_num)] = 999999.9



    for topic_num in range(0, len(topic_model.get_topics())):

        for ind, score in topic_model.get_top_docs(topic=topic_num, n_docs=9999999, sort_by='log_prob'):

            df['topic_' + str(topic_num)].iloc[ind] = score



    # finally save the dataframe so we can load it quicker in situations where we just want to interact with the results.



    pkl.dump(df, open(final_df_filename, "wb"))
# because we are doing our own tokenization, we use a dummy function to bypass

def dummy_fun(doc):

    return doc



tfidf = TfidfVectorizer(

    analyzer='word',

    tokenizer=dummy_fun,

    preprocessor=dummy_fun,

    token_pattern=None)



tfidf_docs = df['clean_tfidf'].tolist()

tfidf_matrix = tfidf.fit_transform(tfidf_docs)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



@interact

def search_articles(

    query='cruise ship spread rate',

    topic=topic_list,

    topic_threshold=(-20, 0, 0.01)

):

    clean_query_words = remove_non_dict_words(get_ngram_words(clean(query)))

    query_vector = tfidf.transform([clean_query_words])

    

    scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    

    df['cosine_dist'] = scores



    # these are the ordered search results according to TF-IDF



    # smaller corex_topic scores means more likely to be of that topic

    corex_cols = [col for col in df if col.startswith('topic_')]

    select_cols = ['title', 'abstract', 'authors', 'cosine_dist'] + corex_cols

    

    results = df[select_cols].loc[df[topic.split(':')[0]] > topic_threshold].sort_values(by=['cosine_dist'], ascending=False).head(10)

    

    top_row = results.iloc[0]

    

    print('TOP RESULT:\n')

    print(top_row['title'] + '\n')

    print(top_row['abstract'])

    

    print('\nAUTHORS:\n')

    print(top_row['authors'])

    

    return results