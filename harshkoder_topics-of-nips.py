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
papers=pd.read_csv('/kaggle/input/nips-papers/papers.csv')

papers.head()
papers.loc[papers['abstract']!='Abstract Missing']

papers.shape[0],papers.loc[papers['abstract']!='Abstract Missing'].shape[0]
papers.drop(['id','event_type','pdf_name'],axis=1,inplace=True)

papers.groupby('year').size().plot(kind='line')


import spacy

nlp = spacy.load('en_core_web_sm')

"""if (t.pos_ == 'ADJ'  or t.pos_=='ADV')"""

def tokenize(s):

    return [t.lemma_ for t in nlp(s.lower())  if t.is_alpha if not t.is_stop]

corpus_01=papers['title'].apply(tokenize)

" ".join(corpus_01[0])
from gensim.models import Phrases

bigram = Phrases(corpus_01)

grammer=lambda x:bigram[x]
corpus_02=corpus_01.apply(grammer)
trigram=Phrases(corpus_02)

grammer=lambda x:trigram[x]

corpus_03=corpus_02.apply(grammer)
" ".join(corpus_03[1])
quadgrams=Phrases(corpus_03)

grammer=lambda x:quadgrams[x]

corpus_04=corpus_03.apply(grammer)
from gensim import corpora

dictionary = corpora.Dictionary(corpus_04)

corpus_05 = [dictionary.doc2bow(text) for text in corpus_04]
from gensim.models.ldamodel import LdaModel

lda_model = LdaModel(corpus=corpus_05,id2word=dictionary,num_topics=10)
lda_model.print_topics()
import pyLDAvis

import pyLDAvis.gensim

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus_05, dictionary)

vis