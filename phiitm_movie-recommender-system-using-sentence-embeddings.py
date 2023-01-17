import numpy as np

import pandas as pd

import os

from tqdm import tqdm_notebook

import tensorflow as tf

import tensorflow_hub as hub

from nltk import sent_tokenize

from tqdm import tqdm

from scipy import spatial

from operator import itemgetter

tqdm.pandas()

%matplotlib inline
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
movie = pd.read_csv('../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')

full_mov = pd.read_csv('../input/movie-database/full_mov_db.csv')

full_mov.head()
movie.head()
len(movie)
## Drop duplicates

movie = movie.drop_duplicates(subset='Plot', keep="first")

len(movie)
movie.reset_index(inplace=True)

movie.drop(columns=['index'],inplace=True)

movie.head()
movie['word count'] = movie['Plot'].apply(lambda x : len(x.split()))
movie['word count'].iplot(

    kind='hist',

    bins=100,

    xTitle='word count',

    linecolor='black',

    yTitle='no of plots',

    title='Plot Word Count Distribution')
movie['Origin/Ethnicity'].value_counts().iplot(kind='bar')
movie['Release Year'].value_counts().iplot(kind='bar')
import re

def clean_plot(text_list):

    clean_list = []

    for sent in text_list:

        sent = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-.:;<=>?@[\]^`{|}~"""), '',sent)

        sent = sent.replace('[]','')

        sent = re.sub('\d+',' ',sent)

        sent = sent.lower()

        clean_list.append(sent)

    return clean_list
plot_emb_list = []

with tf.Graph().as_default():

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    messages = tf.placeholder(dtype=tf.string, shape=[None])

    output = embed(messages)

    with tf.Session() as session:

        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for plot in tqdm_notebook(full_mov['Plot']):

            sent_list = sent_tokenize(plot)

            clean_sent_list = clean_plot(sent_list)

            sent_embed = session.run(output, feed_dict={messages: clean_sent_list})

            plot_emb_list.append(sent_embed.mean(axis=0).reshape(1,512))            

full_mov['embeddings'] = plot_emb_list

full_mov.head()
full_mov.to_pickle('./updated_mov_df_use_2.pkl')
def similar_movie(movie_name,topn=5):

    plot = full_mov.loc[movie_name,'Plot'][:1][0]

    with tf.Graph().as_default():

        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

        messages = tf.placeholder(dtype=tf.string, shape=[None])

        output = embed(messages)

        with tf.Session() as session:

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])

            sent_list = sent_tokenize(plot)

            clean_sent_list = clean_plot(sent_list)

            sent_embed2 = (session.run(output, feed_dict={messages: clean_sent_list})).mean(axis=0).reshape(1,512)

            similarities = []

            for tensor,title in zip(full_mov['embeddings'],full_mov.index):

                cos_sim = 1 - spatial.distance.cosine(sent_embed2,tensor)

                similarities.append((title,cos_sim))

            return sorted(similarities,key=itemgetter(1),reverse=True)[1:topn+1]
full_mov.set_index('Title', inplace=True)
similar_movie('Interstellar')