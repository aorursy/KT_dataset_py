import numpy as np

import pandas as pd

import json

import os



datafiles = []

for dirname, _, filenames in os.walk("/kaggle/input/CORD-19-research-challenge/"):

    for filename in filenames:

        ifile = os.path.join(dirname, filename)

        if ifile.split(".")[-1] == "json":

            datafiles.append(ifile)

doc_ids = []

titles = []

abstracts = []

bodytexts = []

id2title = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    doc_ids.append(doc['paper_id']) 

    titles.append(doc['metadata']['title'])

    abstract = ''

    for item in doc['abstract']:

        abstract = abstract + item['text']

    abstracts.append(abstract)

    bodytext = ''

    for item in doc['body_text']:

        bodytext = bodytext + item['text']

    bodytexts.append(bodytext)

texts = np.array([t + ' ' + a for t, a in zip(titles, abstracts)], dtype='object')
from nltk.tokenize import word_tokenize

from gensim.sklearn_api.phrases import PhrasesTransformer



bigrams = PhrasesTransformer(min_count=20, threshold=100)

processed_texts = bigrams.fit_transform([word_tokenize(text) for text in texts])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation



vectorizer = CountVectorizer(stop_words="english", lowercase=True, max_df=0.1, min_df=20)

dtm = vectorizer.fit_transform([" ".join(text) for text in processed_texts])

k = 8

lda = LatentDirichletAllocation(n_components=k, 

                                learning_method="batch", 

                                doc_topic_prior=1/k,

                                topic_word_prior=0.1,

                                n_jobs=-1,

                                random_state=0)

lda.fit(dtm)
from pyLDAvis import enable_notebook, display, sklearn



enable_notebook()

topic_model_info = sklearn.prepare(lda, dtm, vectorizer, mds='PCoA')

display(topic_model_info)