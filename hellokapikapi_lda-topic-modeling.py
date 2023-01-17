# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

from gensim import corpora, models

import pandas as pd

import pyLDAvis.gensim

import nltk

from nltk import pos_tag

from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords 

import string

from gensim.models import CoherenceModel

import math

from nltk.corpus import wordnet

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/voted-kaggle-dataset/voted-kaggle-dataset.csv')

df.head()
df.info()
descriptions = df[df['Description'].notnull()]['Description']

descriptions.head()
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):

        return wordnet.ADJ

    elif treebank_tag.startswith('V'):

        return wordnet.VERB

    elif treebank_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
documents = []

texts = []

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english')).union({'data', 'dataset', 'model'})

digits = set(string.digits)

for description in descriptions.iteritems():

    document_str = str(description[1]).lower()

    document_str = document_str.translate(str.maketrans('', '', string.punctuation))

    token_words = word_tokenize(document_str)

    pos_tagged = pos_tag(token_words)

    tokens = [(token, get_wordnet_pos(tag)) for token, tag in pos_tagged]

    lemma_tokens = [lemmatizer.lemmatize(token, tag) for token, tag in tokens]

    document = [w for w in lemma_tokens if (not w in stop_words) and (not w.isdigit()) ]

    documents.append(document)

    texts.append(' '.join(document))

print(documents[0])
dictionary = corpora.Dictionary(documents)



document_term = [dictionary.doc2bow(document) for document in documents]
ldamodel = models.ldamodel.LdaModel(document_term, num_topics=20, id2word = dictionary)

ldamodel.print_topics(num_words=5)
pyLDAvis.enable_notebook()

pyLDAvis.gensim.prepare(ldamodel, document_term, dictionary)

# pyLDAvis.display(visualization)
log_perplexity = ldamodel.log_perplexity(document_term)

print('Perplexity:', math.exp(log_perplexity))
coherence_model_lda = CoherenceModel(model=ldamodel, texts=documents, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('Coherence score:', coherence_lda)