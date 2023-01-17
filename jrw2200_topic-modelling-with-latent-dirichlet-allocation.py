import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style

style.use('ggplot')



import os

from datetime import datetime



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True)



from gensim import corpora, models, similarities



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/OnceUponATimeInHollywood.csv')

df.head()
df['EST'] = df['comment_created_utc'].apply(lambda x: datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
trace = go.Histogram(

    x=df['EST'],

    marker=dict(

        color='blue'

    ),

    opacity=0.75

)



layout = go.Layout(

    title='Comment Activity',

    height=450,

    width=1200,

    xaxis=dict(

        title='Month'

    ),

    yaxis=dict(

        title='Comment Quantity'

    ),

    bargap=0.2,

)



data= [trace]



fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
corpus = df.comment_body.values

corpus[0:5]
from sklearn.feature_extraction.text import CountVectorizer



def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



common_words = get_top_n_words(corpus, 75)

for word, freq in common_words:

    pass

    #print(word, freq)

df1 = pd.DataFrame(common_words, columns = ['word' , 'count'])
import plotly.express as px

data = px.data.gapminder()



fig = px.bar(df1.iloc[:50], x='word', y='count', labels={'word':'Word'}, height=400)

fig.show()
fig = px.bar(df1.iloc[50:], x='word', y='count', labels={'word':'Word'}, height=400)

fig.show()
l1 = list(df1.word.values)

keep = ['tarantino','sharon','rick','movies','know','cliff','tate','scene','manson']

[l1.remove(w) for w in keep] 

l1 = l1[:-8]
import gensim

import logging

import tempfile



TEMP_FOLDER = tempfile.gettempdir()

print('We will save the temporary dictionary and corpus at this location: {}'.format(TEMP_FOLDER))



from gensim import corpora

logging.basicConfig(format = '%(ascitime)s: %(levelname)s: %(message)s', level=logging.INFO)
from nltk.corpus import stopwords

from string import punctuation



stoplist = stopwords.words('english') + list(punctuation) + l1



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]
dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'onceUponATime.dict'))

print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER,  'onceUponATime.dict'), corpus)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
total_topics = 3
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf]
lda.show_topics(total_topics,4)
from collections import OrderedDict



data_lda = {i: OrderedDict(lda.show_topic(i, 12)) for i in range(total_topics)}

#data_lda
df_lda = pd.DataFrame(data_lda)

print(df_lda.shape)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
g = sns.clustermap(df_lda.corr(), center=0, cmap='RdBu', metric='cosine', linewidth=0.75, annot_kws={"size": 24}, figsize=(12, 12))

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

fig = plt.gcf()

sns.set(font_scale=2.0)

fig.set_size_inches(18, 18)

plt.savefig('lda_clustermap_tarantino.png')

plt.show()
import pyLDAvis.gensim



pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

pyLDAvis.save_html(panel, 'lda.html')

panel