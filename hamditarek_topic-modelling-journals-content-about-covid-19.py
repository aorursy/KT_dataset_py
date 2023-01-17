import os

import json

import numpy as np 

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import re

from IPython.display import display

from tqdm import tqdm

from collections import Counter

import ast

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



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sb



from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.manifold import TSNE



from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

output_notebook()



%matplotlib inline



import os

        

from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm_notebook as tqdm

from Levenshtein import ratio as levenshtein_distance



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text



from scipy import spatial
## load the data 

df_train = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
#Looking data format and types

print(df_train.info())
#Take a look at the data

df_train.head(n=2)
df_train["title"].head()
q = df_train["title"].to_list()

for i in range(20):

    print('Article title '+str(i+1)+': '+q[i])
# For the Abstract

q = df_train["abstract"].to_list()

q = [x for x in q if str(x) != 'nan']

j=0

for i in q[:20]:

    print('Abstract content '+str(j+1)+': '+i)
# Plotting a bar graph of the number of stores in each city, for the first ten cities listed

# in the column 'City'

journal_count  = df_train['journal'].value_counts()

journal_count = journal_count[:10,]

plt.figure(figsize=(10,5))

sns.barplot(journal_count.index, journal_count.values, alpha=0.8)

plt.title('Top 10 journal published about COVID-19')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('journal', fontsize=12)

plt.show()
# For the Abstract

reindexed_data = df_train["abstract"].dropna()
# Define helper functions

def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii').decode('utf-8') for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data)



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
reindexed_data1 = df_train["journal"].dropna()
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data1)



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
reindexed_data2 = df_train["authors"].dropna()



count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data2)



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
# For the Abstract

reindexed_data = df_train["abstract"].dropna().tolist()

tagged_headlines = [TextBlob(reindexed_data[i]).pos_tags for i in range(len(reindexed_data))]
tagged_headlines_df = pd.DataFrame({'tags':tagged_headlines})



word_counts = [] 

pos_counts = {}



for headline in tagged_headlines_df[u'tags']:

    word_counts.append(len(headline))

    for tag in headline:

        if tag[1] in pos_counts:

            pos_counts[tag[1]] += 1

        else:

            pos_counts[tag[1]] = 1

            

print('Total number of words: ', np.sum(word_counts))

print('Mean number of words per Abstract: ', np.mean(word_counts))
y = stats.norm.pdf(np.linspace(0,14,50), np.mean(word_counts), np.std(word_counts))



fig, ax = plt.subplots(figsize=(8,4))

ax.hist(word_counts, bins=range(1,14), density=True);

ax.plot(np.linspace(0,14,50), y, 'r--', linewidth=1);

ax.set_title('Abstract word lengths');

ax.set_xticks(range(1,14));

ax.set_xlabel('Number of words');

plt.show()
pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)

pos_sorted_counts = sorted(pos_counts.values(), reverse=True)



fig, ax = plt.subplots(figsize=(14,4))

ax.bar(range(len(pos_counts)), pos_sorted_counts);

ax.set_xticks(range(len(pos_counts)));

ax.set_xticklabels(pos_sorted_types);

ax.set_title('Part-of-Speech Tagging for questions Corpus');

ax.set_xlabel('Type of Word');
reindexed_data = df_train["abstract"].dropna()

small_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)

small_text_sample = reindexed_data.sample(n=500, random_state=0).values



print('Abstracts before vectorization: {}'.format(small_text_sample[123]))



small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)



print('Abstracts after vectorization: \n{}'.format(small_document_term_matrix[123]))
#number of topics

n_topics = 5
lsa_model = TruncatedSVD(n_components=n_topics)

lsa_topic_matrix = lsa_model.fit_transform(small_document_term_matrix)
# Define helper functions

def get_keys(topic_matrix):

    '''

    returns an integer list of predicted topic 

    categories for a given topic matrix

    '''

    keys = topic_matrix.argmax(axis=1).tolist()

    return keys



def keys_to_counts(keys):

    '''

    returns a tuple of topic categories and their 

    accompanying magnitudes for a given list of keys

    '''

    count_pairs = Counter(keys).items()

    categories = [pair[0] for pair in count_pairs]

    counts = [pair[1] for pair in count_pairs]

    return (categories, counts)
lsa_keys = get_keys(lsa_topic_matrix)

lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
# Define helper functions

def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):

    '''

    returns a list of n_topic strings, where each string contains the n most common 

    words in a predicted category, in order

    '''

    top_word_indices = []

    for topic in range(n_topics):

        temp_vector_sum = 0

        for i in range(len(keys)):

            if keys[i] == topic:

                temp_vector_sum += document_term_matrix[i]

        temp_vector_sum = temp_vector_sum.toarray()

        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)

        top_word_indices.append(top_n_word_indices)   

    top_words = []

    for topic in top_word_indices:

        topic_words = []

        for index in topic:

            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))

            temp_word_vector[:,index] = 1

            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]

            topic_words.append(the_word.encode('ascii').decode('utf-8'))

        top_words.append(" ".join(topic_words))         

    return top_words
tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 

                        n_iter=2000, verbose=1, random_state=0, angle=0.75)

tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)
# Define helper functions

def get_mean_topic_vectors(keys, two_dim_vectors):

    '''

    returns a list of centroid vectors from each predicted topic category

    '''

    mean_topic_vectors = []

    for t in range(n_topics):

        articles_in_that_topic = []

        for i in range(len(keys)):

            if keys[i] == t:

                articles_in_that_topic.append(two_dim_vectors[i])    

        

        articles_in_that_topic = np.vstack(articles_in_that_topic)

        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)

        mean_topic_vectors.append(mean_article_in_that_topic)

    return mean_topic_vectors
colormap = np.array([

    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",

    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",

    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",

    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])

colormap = colormap[:n_topics]
# Preparing a corpus for analysis and checking the first 5 entries

corpus=[]



corpus = df_train["abstract"].dropna().to_list()



corpus[:2]
TEMP_FOLDER = tempfile.gettempdir()

print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing common words and tokenizing

# google-quest-challenge

stoplist = stopwords.words('english') + list(punctuation) + list("([)]?") + [")?"]



texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]



dictionary = corpora.Dictionary(texts)

dictionary.save(os.path.join(TEMP_FOLDER, 'google-quest-challenge.dict'))  # store the dictionary,
corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'google-quest-challenge.mm'), corpus) 
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
#I will try 15 topics

total_topics = 15



lda = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics)

corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tf
lda.show_topics(total_topics,5)
data_lda = {i: OrderedDict(lda.show_topic(i,25)) for i in range(total_topics)}
df_lda = pd.DataFrame(data_lda)

df_lda = df_lda.fillna(0).T

print(df_lda.shape)
df_lda
g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="OrRd", metric='cosine', linewidths=.75, figsize=(12, 12))

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()

#plt.setp(ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='tsne')

panel