import os

import pandas as pd

from gensim import corpora, models, similarities
datafile = '../input/bias-media-cat/all3.csv'
import pandas as pd

tweets = pd.read_csv(datafile, encoding='utf8')

#// tweets = tweets.assign(Time=pd.to_datetime(tweets.time)).drop('tweet_id', axis='columns')

#x = ""+tweets.year+"/"+tweets.month + "/" +tweets.date +" " +tweets.time

#tweets = tweets.assign(Time=pd.to_datetime(x)).drop('tweet_id', axis='columns')



tweets.head(10)
range(len(tweets['tweet']))



range( len(tweets[tweets['tweet'].str.contains("atal|Catalonia|Catalunya|Cataluña")==True]['tweet'] ))
corpus=[]

a=[]

# remove punctiation

tweets["tweet"] = tweets['tweet'].str.replace('[^\w\s]','')

tweets["tweet"] = tweets["tweet"] + " " + tweets["screen_name"] 

for a in tweets[tweets['tweet'].str.contains("Catalonia|Catalunya|Cataluña")==True]['tweet']:

        corpus.append(a)
corpus[0:5]
import gensim

import logging

import tempfile



TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.corpus import stopwords

from string import punctuation



# remove common words and tokenize

list1 = ['RT','rt','es','el','per','amb','pel','parter','directo','és','els','tras','minuto','hora','social']

stoplist =  stopwords.words('spanish') + stopwords.words('english') + list(punctuation) + list1



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]
dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'elon.dict'))  # store the dictionary, for future reference

#print(dictionary)
#print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'elon.mm'), corpus)  # store to disk, for later use
from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
total_topics = 5
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#Show first n important word in the topics:

lda.show_topics(total_topics,5)
from collections import OrderedDict



data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}

#data_lda
import pandas as pd



df_lda = pd.DataFrame(data_lda)

print(df_lda.shape)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



g=sns.clustermap(df_lda.corr(), center=0, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(12, 12))

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()

#plt.setp(ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
import pyLDAvis.gensim



pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel