from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise        import cosine_similarity

from sklearn.cluster                 import DBSCAN



from nltk.stem     import WordNetLemmatizer,PorterStemmer

from nltk.tokenize import word_tokenize

from nltk.corpus   import stopwords



from collections   import Counter

from tqdm          import tqdm



import matplotlib.pyplot as plt

import seaborn           as sns

import pandas            as pd

import numpy             as np



import pyLDAvis.gensim

import pyLDAvis

import gensim

import spacy

import os

import os





os.listdir('../input/CORD-19-research-challenge/2020-03-13')
data_path = '../input/CORD-19-research-challenge/2020-03-13'

sources = pd.read_csv(os.path.join(data_path, 'all_sources_metadata_2020-03-13.csv'))
sources.head()
'''

    Valores nulos

'''

sources.isna().sum()

stop = set(stopwords.words('english'))



def build_list(df, col):

    lemmatizer = WordNetLemmatizer()

    new_df     = df[col].dropna().str.split()

    new_df     = new_df.values.tolist()

    corpus     = [lemmatizer.lemmatize(word.lower()) for i in new_df for word in i if(word) not in stop]

    

    return corpus
def get_x_y(most_common):

    x = []

    y = []

    for word, count in most_common:

        if (word not in stop):

            x.append(word)

            y.append(count)

    

    return x, y
def plot_most_common_words_from(col, number_of_words=25):

    corpus      = build_list(sources, col)

    counter     = Counter(corpus)

    most_common = counter.most_common()



    x, y = get_x_y(most_common[:number_of_words])

    

    plt.figure(figsize=(9,7))

    sns.barplot(x=y,y=x)
def get_top_ngram(corpus, n=None):

    vectorizer   = CountVectorizer(ngram_range=(n, n)).fit(corpus)

    bag_of_words = vectorizer.transform(corpus)

    sum_words    = bag_of_words.sum(axis=0) 

    words_freq   = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

    words_freq   = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:10]
def plot_top_ngram(df, col, n=2, number_of_ngrams=25):

    corpus        = df[col].dropna()

    top_n_ngrams = get_top_ngram(corpus, n)[:number_of_ngrams]

    

    x, y = map(list,zip(*top_n_ngrams))

    

    plt.figure(figsize=(9,7))

    sns.barplot(x=y,y=x)
plot_most_common_words_from('title')
plot_most_common_words_from('abstract')
plot_top_ngram(sources, 'title', 2)
plot_top_ngram(sources, 'abstract', 2)
plot_top_ngram(sources, 'title', 3)
plot_top_ngram(sources, 'abstract', 3)