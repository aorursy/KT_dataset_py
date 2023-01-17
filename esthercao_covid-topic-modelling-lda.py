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
datafile = '/kaggle/input/coronavirus-covid19-tweets-late-april/2020-04-16 Coronavirus Tweets.CSV'
# read tweets, filter to USA and English, sample to 1000



tweets1 = pd.read_csv(datafile, encoding='latin1')

tweets1 =tweets1[(tweets1.country_code == "US") & (tweets1.lang == "en")].reset_index(drop = True)

tweets2 = tweets1[['text']].copy()

tweets2 = tweets2.rename(columns={'text': 'Tweet'})

tweets = tweets2.sample(1000).reset_index()



print("Number of tweets: ",len(tweets['Tweet']))

tweets.head(5)
# Preparing a corpus for analysis and checking first 10 entries



corpus=[]

a=[]

for i in range(len(tweets['Tweet'])):

        a=tweets['Tweet'][i]

        corpus.append(a)

        

corpus[0:10]
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing. Also removing Covid hash tags, as it appeared in all LDA topics equally. 



list1 = ['RT','rt','#coronavirus','#covid19','#covid_19']

stoplist = stopwords.words('english') + list(punctuation) + list1



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]



dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict'))  # store the dictionary, for future reference
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'elon.mm'), corpus)  # store to disk, for later use
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
total_topics = 8



lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#Show first n important word in the topics:

lda.show_topics(total_topics,10)
for topic_id in range(lda.num_topics):

    topk = lda.show_topic(topic_id, 30)

    topk_words = [ w for w, _ in topk ]

    print('{}: {}'.format(topic_id, ' '.join(topk_words)))
#Add top topic and score for each document



tweets['topic1'] = np.nan

tweets['score1'] = np.nan

for i in range(len(tweets['Tweet'])):

    topic = lda.get_document_topics(lda.id2word.doc2bow(tweets['Tweet'][i].split()))[0]

    tweets['topic1'][i] = topic[0]

    tweets['score1'][i] = topic[1]



# Save to file to be downloaded



dfTopic = tweets.sort_values(by=['score1'], ascending = False).head(500)

dfTopic = dfTopic.sort_values(by=['topic1'])

dfTopic.to_csv('/kaggle/working/topics.csv',index=False)

dfTopic.head(50)

data_lda = {i: OrderedDict(lda.show_topic(i,10)) for i in range(total_topics)}
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