import os

import pandas as pd

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

import gensim

from gensim import corpora, models, similarities

import logging

import tempfile

from nltk.corpus import stopwords

from string import punctuation

from collections import OrderedDict

import seaborn as sns

import pyLDAvis.gensim

import matplotlib.pyplot as plt

%matplotlib inline



init_notebook_mode(connected=True) #do not miss this line



import warnings

warnings.filterwarnings("ignore")
questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv', low_memory=False)
questions.head()
# Preparing a corpus for analysis and checking all entries

col = questions.columns[1:]

corpus=[]

a=[]

for q in col:

        a=questions[q][0]

        corpus.append(a)

        

corpus
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing

stoplist = stopwords.words('english') + list(punctuation) + list("([)]?") + [")?"]



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]



dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'kaggle_survey.dict'))  # store the dictionary, for future reference
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'kaggle_survey.mm'), corpus)  # store to disk, for later use
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#Show first n important word in the topics:

lda.show_topics(total_topics,5)
data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}

#data_lda
df_lda = pd.DataFrame(data_lda)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()

#plt.setp(ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel