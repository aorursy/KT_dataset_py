import os
print(os.listdir("../input"))
# https://www.kaggle.com/jieyang0311/covid-19-topic-modeling-lda
# load library
import os
import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import datetime
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nltk
# Load the metadata.csv
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
print(meta.shape)
# First filter the meta file. Select only papers between 1st January and 31st March
meta["publish_time"] = pd.to_datetime(meta["publish_time"])
meta["publish_year"] = (pd.DatetimeIndex(meta['publish_time']).year)
meta["publish_month"] = (pd.DatetimeIndex(meta['publish_time']).month)
meta["publish_day"] = (pd.DatetimeIndex(meta['publish_time']).day)
meta = meta[meta["publish_year"] == 2020]
meta = meta[meta["publish_month"] <= 3]
print(meta.shape[0], " papers are available between 2020 Jan 1 and 2020 March 31.")
# Sort the meta using publish time
meta = meta.sort_values(["publish_time"]);
meta[-3:]
#count how many has abstract
count = 0
index = []
for i in range(len(meta)):
    #print(i)
    if type(meta.iloc[i, 8])== float:
        count += 1
    else:
        index.append(i)

print(len(index), " papers have abstract available.")
meta = meta.iloc[index]
len(meta)
##extract the abstract to pandas 
documents = meta.iloc[:, 8]
documents = documents.reset_index()
documents.drop("index", inplace = True, axis = 1)

##create pandas data frame with all abstracts, use as input corpus
documents["index"] = documents.index.values
documents.head(3)
##extract the publication time to pandas
pub_time = meta.iloc[:, 9]
pub_time = pub_time.reset_index()
pub_time.drop("index", inplace = True, axis = 1)

##create pandas data frame with all abstracts, use as input corpus
pub_time["index"] = pub_time.index.values
pub_time.head(3)
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language

import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


###################################
#### sentence level preprocess ####
###################################

# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()


# language detection
def f_lan(s):
    """
    :param s: string to be processed
    :return: boolean (s is English)
    """

    # some reviews are actually english but biased toward french
    return detect_language(s) in {'English', 'French','Spanish','Chinese'}


###############################
#### word level preprocess ####
###############################

# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


# typo correction
def f_typo(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with typo fixed by symspell. words with no match up will be dropped
    """
    w_list_fixed = []
    for word in w_list:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
        if suggestions:
            w_list_fixed.append(suggestions[0].term)
        else:
            pass
            # do word segmentation, deprecated for inefficiency
            # w_seg = sym_spell.word_segmentation(phrase=word)
            # w_list_fixed.extend(w_seg.corrected_string.split())
    return w_list_fixed


# stemming if doing word-wise
p_stemmer = PorterStemmer()


def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    return [p_stemmer.stem(word) for word in w_list]


# filtering out stop words
# create English stop words list

stop_words = (list(
    set(get_stop_words('en'))
    |set(get_stop_words('es'))
    |set(get_stop_words('de'))
    |set(get_stop_words('it'))
    |set(get_stop_words('ca'))
    #|set(get_stop_words('cy'))
    |set(get_stop_words('pt'))
    #|set(get_stop_words('tl'))
    |set(get_stop_words('pl'))
    #|set(get_stop_words('et'))
    |set(get_stop_words('da'))
    |set(get_stop_words('ru'))
    #|set(get_stop_words('so'))
    |set(get_stop_words('sv'))
    |set(get_stop_words('sk'))
    #|set(get_stop_words('cs'))
    |set(get_stop_words('nl'))
    #|set(get_stop_words('sl'))
    #|set(get_stop_words('no'))
    #|set(get_stop_words('zh-cn'))
))





def f_stopw(w_list):
    """
    filtering out stop words
    """
    return [word for word in w_list if word not in stop_words]


def preprocess_sent(rw):
    """
    Get sentence level preprocessed data from raw review texts
    :param rw: review to be processed
    :return: sentence level pre-processed review
    """
    s = f_base(rw)
    if not f_lan(s):
        return None
    return s


def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, fix typo, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed review
    """
    if not s:
        return None
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_typo(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list


def preprocess(docs, samp_size=None):
    """
    Preprocess the data
    """
    if not samp_size:
        samp_size = 100
    if samp_size > len(docs):
        samp_size = len(docs)

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
    idx_in = []  # index of sample selected
    #     samp = list(range(100))
    samp = np.random.choice(n_docs, samp_size)
    samp.sort()
    for i, idx in enumerate(samp):
        sentence = preprocess_sent(docs[idx])
        token_list = preprocess_word(sentence)
        if token_list:
            idx_in.append(idx)
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in

data = documents
data = data.fillna('')
rws = data.abstract
samp_size = 51000
sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size)
import pickle

dictionary = corpora.Dictionary(token_lists)
bow_corpus = [dictionary.doc2bow(text) for text in token_lists]

tfidf_model = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf_model[bow_corpus]

pickle.dump(dictionary, open('dictionary.pkl', 'wb'))
pickle.dump(bow_corpus, open('bow_corpus.pkl', 'wb'))
pickle.dump(tfidf_corpus, open('tfidf_corpus.pkl', 'wb'))
from gensim.models import LdaSeqModel
import pickle

dictionary = pickle.load(open('dictionary.pkl', 'rb'))
tfidf_corpus = pickle.load(open('tfidf_corpus.pkl', 'rb'))

num_slice = 6
num_topics = 5
num_docs_per_slice = int(len(tfidf_corpus)/num_slice)
time_slice = [num_docs_per_slice for _ in range(num_slice-1)]
time_slice.append(len(tfidf_corpus)-(num_slice-1)*num_docs_per_slice)

ldaseq = LdaSeqModel(corpus=tfidf_corpus, time_slice=time_slice, num_topics=num_topics, id2word=dictionary)
ldaseq.save("ldaseq_tfidf_model_"+str(num_topics)+'_'+str(num_slice))
from gensim.models import LdaSeqModel
ldaseq = LdaSeqModel.load("ldaseq_tfidf_model_5_6")
# Observing the evolution of topic distribution

import pandas as pd

def get_topic_dist_time(model, top_terms=10):
    """
    For each topic obtain the top terms in all the time slice
    :param model: the trained LdaSeqModel
    :param top_terms: the number of top terms to be obtained
    :return: the list of panda dataframe containing the top terms
             of all the topics
    """
    
    df_list = []
    for i in range(model.num_topics):
        word_list = []
        for j in range(6):
            words = ldaseq.print_topic(i, j, top_terms=top_terms)
            word_list.append(words)
        keys = ['Jan 01-Jan 15', 'Jan 15-Jan 31', 'Feb 01-Feb 15', \
                'Feb 15-Feb 29', 'Mar 01-Mar 15', 'Mar 15-Jan 31']
        dic = {}
        for i in range(6):
            dic[keys[i]] = []
        for k in range(top_terms):
            for j in range(6):
                w, p = word_list[j][k]
                dic[keys[j]].append(w)
        df_list.append(pd.DataFrame(dic))
    return df_list

df_list = get_topic_dist_time(ldaseq, 15)
print("Topic 1: top 15 terms")
df_list[0]
print("Topic 2: top 15 terms")
df_list[1]
print("Topic 3: top 15 terms")
df_list[2]
print("Topic 4: top 15 terms")
df_list[3]
print("Topic 5: top 15 terms")
df_list[4]
# Volume of research on each topic in different time period
# Print the marginal probability of all the topics

import numpy as np

def get_doc_sizes(corpus):
    """
    Computes the size of each documents of the corpus
    :param corpus: the text corpus (bag-of-words or tf-idf)
    :return: the list of document sizes
    """
    doc_sizes = []
    for doc in bow_corpus:
        size = sum([v for k, v in doc])
        doc_sizes.append(size)
    return doc_sizes
    
def get_marginal_topic_dist(docs, doc_sizes, model):
    """
    Computes the marginal probability of a list of the documents 
    :param docs: list of document ids
    :param doc_sizes: the list of sizes of the documents in docs
    :param model: the trained LdaSeqModel model
    :return: the marginal probabilities of documents as list
    """
    prob = np.zeros((model.num_topics))
    for doc in docs:
        prob += np.array(model.doc_topics(doc)) * doc_sizes[doc]
    corpus_size = sum([doc_sizes[d] for d in docs])
    prob /= corpus_size
    return list(prob)

def get_time_slice(num_docs, num_slices):
    """
    Computes the number of documents in each time slice
    :param num_docs: the total number of documents
    :param num_slices: the number of total slices
    :return: the list of the number of documents in each time slice
    """
    num_docs_per_slice = int(num_docs/num_slices)
    time_slice = [num_docs_per_slice for _ in range(num_slices-1)]
    time_slice.append(num_docs-(num_slices-1)*num_docs_per_slice)
    return time_slice
    
ldaseq = LdaSeqModel.load("ldaseq_tfidf_model_5_6")
tfidf_corpus = pickle.load(open('tfidf_corpus.pkl', 'rb'))
doc_sizes = get_doc_sizes(tfidf_corpus)

time_slice = get_time_slice(len(tfidf_corpus), 6)
st_index = 0
marg_dist_list = []
for s in time_slice:
    dist = get_marginal_topic_dist(range(st_index, st_index+s), doc_sizes, ldaseq)
    marg_dist_list.append(dist)
    st_index += s
# Plotting the pie chart for the total volume of research in each topic

import matplotlib.pyplot as plt

labels = 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'

# Plots the pie chart for the time slice Jan 1 - Jan 15
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))
ax1.pie(marg_dist_list[0], labels=labels, autopct='%1.2f%%', textprops={'fontsize': 14})
ax1.axis('equal')
ax1.set_title('Jan 1 - Jan 15', fontsize=25, pad=35)

# Plots the pie chart for the time slice Feb 15 - Feb 29
ax2.pie(marg_dist_list[3], labels=labels, autopct='%1.2f%%', textprops={'fontsize': 14})
ax2.axis('equal')
ax2.set_title('Feb 15 - Feb 29', fontsize=25, pad=35)

# Plots the pie chart for the time slice Mar 15 - Mar 31
ax3.pie(marg_dist_list[5], labels=labels, autopct='%1.2f%%', textprops={'fontsize': 14})
ax3.axis('equal')
ax3.set_title('Mar 15 - Mar 31', fontsize=25, pad=35)

#plt.tight_layout()
plt.show()