# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import gensim
from nltk import WordNetLemmatizer, SnowballStemmer, word_tokenize, sent_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import *
import re
from string import punctuation

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

stemmer = SnowballStemmer("english")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Friday, 3/20/2020; 10:34 PM

# Total Dataset Size: 29315

# biorxiv_medrxiv - 885
bm = len(os.listdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/'))

# noncomm_use_subset - 2353
nus = len(os.listdir('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'))

# comm_use_subset - 9118
cus = len(os.listdir('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'))

# custom_license - 16959
cl = len(os.listdir('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/'))

bm + nus + cus + cl
# os.chdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/')
# os.chdir('/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/')
# os.chdir('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/')
# os.chdir('/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/')

# SplitTypes = []
# for file in os.listdir(os.curdir):
#     SplitTypes.append(file.split('.')[-1])
# print(list(set(SplitTypes)))
# Sunday, 3/22/20; 6:00 PM
os.chdir('/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/')
!cat 0015023cc06b5362d332b3baf348d11567ca2fbb.json
# Monday, 3/23/20; 11:13 PM
DATA_PATH = '/kaggle/input/CORD-19-research-challenge/'

# (?) Function for parsing the date column
# dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
# df = pd.read_csv(DATA_PATH + 'metadata.csv', parse_dates = ['publish_time'], date_parser = dateparse)

df = pd.read_csv(DATA_PATH + 'metadata.csv', sep = ',')

print(df.head())
print()
print(df.shape)
print()
print(df.info())
# EDA Questions of Interest

# What percent of values are present?
print(df.count() / df.shape[0])

# What is the breakdown of the 'has_full_text' field?
print()
print(df['has_full_text'].value_counts())
# Support functions
def remove_stopwords(tokens, stopwords_list = stopwords.words('english'), custom_stopwords = []):
    """Removes the word from a list if it is present in the `stopwords_list`

    Parameters
    ----------
    tokens : list
        List of words / text to process
    stopwords_list : list
        List of words to remove (default is nltk's English stopwords list)

    Returns
    -------
    result : list
        A list of the tokens with the stopwords removed
    """
    result = [t for t in tokens if t.lower() not in stopwords_list + custom_stopwords]
    return result


def remove_spec_chars(text, mode='new_line'):
    """Removes special characters, like punctuation or newlines, from text using \
    regular expressions

    Parameters
    ----------
    text : str
        Text string input to remove characters
    mode : str {'new_line', 'punctuation'}
        Type of method / regular expression category to remove. Options are
        `new_line` to remove ``\\n`` chars or `punctuation` to remove general
        punctuation, including but not limited to numbers and any of the
        following symbols: ``.!"'#$%^&*()-+=``


    Returns
    -------
    str
        The input text string with characters replaced with a single whitespace
    """
    # replaces punctuation
    if mode == 'punctuation':
        regex = r'[^\w\s]'
        return re.sub(regex, ' ', text)
    # replaces newline chars
    else:
        regex = r'[\r\n\t]'
        return re.sub(regex, '', text)
    
def process(text, remove_punctuation = True, remove_newlines = True,
        remove_numbers = True, lowercase = True, delete_stopwords = True,
        custom_stopwords = [], lemmatize = True, lemmatizer_source = "wordnet",
        min_word_length = 2):
    """
    Performs requested cleanup on text
    Returns processed text as a string separated by spaces
    Args:
        text (str): text to process, such as the text attribute of a ctxe_doc object
        remove_punctuation (bool): whether to remove punctuation, e.g. "don't do it, please" -> "don t do it please"
        remove_newlines (bool): whether to remove newline characters ('\n') along with carriage returns ('\r') and tabs/indents ('\t')
        remove_numbers (bool): whether to remove numbers
        lowercase (bool): whether to convert all characters to lowercase, e.g. "CraZY cAses" -> "crazy cases"
        delete_stopwords (bool): whether to perform stopword removal, using the standard English stopwords in addition to any others specified in custom_stopwords argument below
        custom_stopwords (list): if delete_stopwords = True, place to specify additional user-specific stopwords (e.g. delete_stopwords = True and custom_stopwords = ['duck', 'goose'] would cause: "I want to play duck duck goose" -> "want play")
        lemmatize (bool): whether to perform lemmatization on each word
        lemmatizer_source (str): if lemmatize = True, which lemmatizer to use (currently only WordNet implemented)
        min_word_length (int): minimum number of characters for a word to be kept; all with fewer characters would be dropped (e.g. min_word_length = 5 would cause: "Nice to meet you, madam" -> "madam")
    """
    # Replace periods and other punctuations if turned on
    if remove_punctuation:
        text = remove_spec_chars(text, mode='punctuation')
    # Replace newlines if turned on
    if remove_newlines:
        text = remove_spec_chars(text)
    # Replace numbers if turned on
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    # Create list of word tokens
    # make all words lowercase if turned on
    if lowercase:
        words = [word.lower() for word in text.split(" ")]
    else:
        words = [word for word in text.split(" ")]
    # Remove stop words if turned on
    if delete_stopwords:
        words = remove_stopwords(words, custom_stopwords = custom_stopwords)
    # Lemmatize and remove punctuation, if turned on
    # TO-DO: Lemmatizing is very slow. Add parallel processing!
    if lemmatize:
        if lemmatizer_source == "wordnet":
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(
                word) for word in words if word not in punctuation]
        else:
            warnings.warn("Lemmatizer {0} not yet implemented, so no lemmatization will occur.".format(
                lemmatizer_source))
    # Remove words below min word length
    words = [word for word in words if len(word) >= min_word_length]
    return " ".join(words)
    # TO-DO: Do we always want to lemmatize? Should we give options to turn this on / off?
    # or are we saying we have an opinion / recommendation that Lemmatizing should happen automatically
    lemmatizer = WordNetLemmatizer()
    clean_words = [lemmatizer.lemmatize(
        word) for word in words if word not in punctuation]
    # Remove single character words
    clean_words = [word for word in clean_words if len(word) > 1]
    return " ".join(clean_words)
#Processing the abstracts
dfAbstract = df[~df['abstract'].isna()]
dfAbstract.head()
dfAbstract = dfAbstract['abstract'].apply(process)
dfAbstract
# Monday, 3/30/20; 9:16 PM
# Tutorial: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# type(dfAbstract)

def preprocess(text):
    result = []
    
    for token in gensim.utils.simple_preprocess(text):
        result.append(token)
    
    return result

dfAbstract = dfAbstract.apply(preprocess)
dfAbstract[:10]
dictionary = gensim.corpora.Dictionary(dfAbstract)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    
    count += 1
    if count > 10:
        break
len(dictionary)
dictionary.filter_extremes(no_below = 15, no_above = 0.5, keep_n = 100000)
len(dictionary)
bow_corpus = [dictionary.doc2bow(doc) for doc in dfAbstract]
bow_corpus[4310]
bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
print(dfAbstract[4310])
print()
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
print(dfAbstract[4310])
print()
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
unseen_document = 'COVID19 was seen in Wuhan in December 2019 and quickly spread to the rest of the work in 2020.'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
