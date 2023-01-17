# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/itsm-dada/all_tickets.csv")

data = pd.DataFrame(data)
data.info()
data = data.dropna()

data.info()
data.head()
len(data['ticket_type'])

data['ticket_type'].value_counts()

#data.ticket_type.unique()

#data.category.unique()

data.category.value_counts()
data['index'] = data.index

data.head()
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(2018)

import nltk

nltk.download('wordnet')
stemmer = PorterStemmer()

STOPWORDS = STOPWORDS.union(set(['Thanks', 'thanks', 'Hello','hello', 'regard', 'hi','kind','dear']))



def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result
doc_sample = data[data['index'] == 4310].values[0][1]

print('original document body: ')

words = []

for word in doc_sample.split(' '):

    words.append(word)

print(words)

print('\n\n tokenized and lemmatized document: ')

print(preprocess(doc_sample))
processed_docs = data['body'].map(preprocess)

processed_docs[:10]
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0

for k, v in dictionary.iteritems():

    print(k, v)

    count += 1

    if count > 10:

        break
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

bow_corpus[4310]
bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):

    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],

                                                     dictionary[bow_doc_4310[i][0]],

                                                     bow_doc_4310[i][1]))
from gensim import corpora, models

from pprint import pprint



tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]



for doc in corpus_tfidf:

    pprint(doc)

    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):

    print('Topic: {} \nWords: {}'.format(idx, topic))
import matplotlib.pyplot as plt

from wordcloud import WordCloud

for t in range(lda_model.num_topics):

    plt.figure()

    #plt.imshow(WordCloud().fit_words(lda_model.show_topic(t, 200)))

    plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(t, 200))))

    plt.axis("off")

    plt.title("Topic #" + str(t))

    plt.show()
import pyLDAvis.gensim



lda_display = pyLDAvis.gensim.prepare(lda_model, bow_corpus,dictionary, sort_topics=True)

pyLDAvis.display(lda_display)
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=2, workers=4)



for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx, topic))
import matplotlib.pyplot as plt

for t in range(lda_model_tfidf.num_topics):

    plt.figure()

    #plt.imshow(WordCloud().fit_words(lda_model_tfidf.show_topic(t, 200)))

    plt.imshow(WordCloud().fit_words(dict(lda_model_tfidf.show_topic(t, 200))))

    plt.axis("off")

    plt.title("Topic #" + str(t))

    plt.show()
import pyLDAvis.gensim



lda_display = pyLDAvis.gensim.prepare(lda_model_tfidf, bow_corpus,dictionary, sort_topics=True)

pyLDAvis.display(lda_display)
processed_docs[4310]
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):

    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
unseen_document = 'I am facing issue with my internet connection after system upgrade'

bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):

    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))