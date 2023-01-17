import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

nRowsRead = None # specify 'None' if want to read whole file

# septuagint.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/septuagint.csv', delimiter=';', nrows = nRowsRead)

df1 = df1.set_index('RunningNumber: Book: Chapter')

df1.dataframeName = 'septuagint.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
#! apt-get update && apt-get install -y --fix-missing liblapack-dev libswscale-dev pkg-config

!pip install textacy

!python -m nltk.downloader stopwords
from nltk.corpus import stopwords

import textacy

from textacy.vsm import Vectorizer

import spacy.cli 
# the below pip stuff don't work in Kaggle Jupiter as it can not find the installed package. 

#!pip install https://github.com/explosion/spacy-models/releases/download/el_core_news_lg-2.3.0/el_core_news_lg-2.3.0.tar.gz --no-deps #egg=el_core_news_lg==2.3.0 

#!python -m spacy link download el_core_news_lg el || true

spacy.cli.download("el_core_news_md")
import el_core_news_md as language

nlp = language.load()
n_jobs=-1



series1=df1[['Text',]].apply(lambda row: ' '.join(row.values.astype(str)) or 'ERROR' , axis=1)

d1=texts_dict = { key:value for key,value in series1.to_dict().items()}



corpus = textacy.corpus.Corpus(lang=nlp)

corpus.add(data=[tuple([value, {'key':key}],)

        for key,value in d1.items()], batch_size=20, n_process=n_jobs)

    

print(  'n_docs=', corpus.n_docs

       ,'n_sentences=',corpus.n_sents

       ,'n_tokens=', corpus.n_tokens)



vectorizer = textacy.vsm.Vectorizer(tf_type='linear', apply_idf=True, idf_type='smooth', norm='l2')



doc_term_matrix = vectorizer.fit_transform(

   (doc._.to_terms_list(

            ngrams={1,2,3}

            , entities=False

            , normalize='lemma'

            , as_strings=True

            , filter_stops=True

            , filter_punct=True

            , filter_nums=True

            )

        for doc in corpus))

        

frequencies = textacy.vsm.matrix_utils.get_term_freqs(doc_term_matrix)

import numpy as np

terms_freqs = sorted(zip(frequencies,vectorizer.terms_list),reverse=True)

print(terms_freqs[:100])
from textacy.ke import sgrank

for doc in corpus:

  print(sgrank(doc, topn=10))