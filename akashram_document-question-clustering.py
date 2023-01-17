import json

import numpy as np 

import pandas as pd

import re

import os

import random



# For plotting

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})



from tqdm import tqdm_notebook as tqdm



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text

from sklearn.metrics.pairwise import cosine_similarity



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))        
ids = []

ans = []

candidates = []

questions = []



with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl', 'r') as json_file:

    cnt = 0

    for line in tqdm(json_file):

        json_data = json.loads(line)        

        ids.append(str(json_data['example_id']))

        questions.append(json_data['question_text'])

        candidates = json_data['long_answer_candidates']
tr_data = pd.DataFrame()



tr_data['example_id'] = ids

tr_data['question'] = questions
tr_data.head(10)
from nltk.stem.lancaster import LancasterStemmer

from nltk.stem.porter import *

import nltk

from nltk.util import ngrams

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize

import re

import string

from nltk.stem import WordNetLemmatizer



stopword_list = nltk.corpus.stopwords.words('english')

wnl = WordNetLemmatizer()

ps = PorterStemmer()



tokenizer = nltk.RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))



def preprocessing(data):

    txt = data.str.lower().str.cat(sep=' ')

    words = tokenizer.tokenize(txt)

    words = [w for w in words if not w in stop_words]

    return words



def tokenize_text(text):

    tokens = nltk.word_tokenize(text)

    tokens = [token.strip() for token in tokens]

    return tokens
def remove_special_characters(text):

    tokens = tokenize_text(text)

    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))

    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])

    filtered_text = ' '.join(filtered_tokens)

    return filtered_text
def remove_stopwords(text):

    tokens = tokenize_text(text)

    filtered_tokens = [token for token in tokens if token not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)

    return filtered_text
def keep_text_characters(text):

    filtered_tokens = []

    tokens = tokenize_text(text)

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    filtered_text = ' '.join(filtered_tokens)

    return filtered_text
def normalize_corpus(corpus, lemmatize=True,  only_text_chars=False, tokenize=False):

    normalized_corpus = []

    for text in corpus:

        text = text.lower()

        text = remove_special_characters(text)

        text = remove_stopwords(text)

        if only_text_chars:

            text = keep_text_characters(text)

 

        if tokenize:

            text = tokenize_text(text)

            normalized_corpus.append(text)

        else:

            normalized_corpus.append(text)

    return normalized_corpus
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def build_feature_matrix(documents, feature_type='frequency',  ngram_range=(1, 1), min_df=0.0, max_df=1.0):

    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':

        vectorizer = CountVectorizer(binary=True, min_df=min_df, max_df=max_df, ngram_range=ngram_range)

    elif feature_type == 'frequency':

        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df, ngram_range=ngram_range)

    elif feature_type == 'tfidf':

        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)

    else:

        raise Exception("Wrong feature type. Possible values are binary, frequency, or tfidf")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix
## Taking only a subset of questions



qns = tr_data['question'][:500]
norm_docs = normalize_corpus(qns,  lemmatize=True, only_text_chars=True)
tr_data_500 = tr_data.iloc[:500]
vectorizer, feature_matrix = build_feature_matrix(norm_docs, feature_type='tfidf', min_df=0, max_df=0.8, ngram_range=(1, 1))
print(feature_matrix.shape)
# get feature names

feature_names = vectorizer.get_feature_names()
# print sample features

print(feature_names[:20])
from sklearn.cluster import KMeans



# define the k-means clustering function



def k_means(feature_matrix, num_clusters=5):

    km = KMeans(n_clusters=num_clusters, max_iter=10000)

    km.fit(feature_matrix)

    clusters = km.labels_

    return km, clusters
# set k = 10(decided arbitrarily, right approach would be elbow method/silhoutte score which we will get to).

# Lets say we want 10 clusters from the list of questions we got 



num_clusters = 10

km_obj, clusters = k_means(feature_matrix=feature_matrix,num_clusters=num_clusters)

tr_data_500['Clusters'] = clusters
from collections import Counter



## Getting the total questions per cluster 



c = Counter(clusters)

print(c.items())
def get_cluster_data(clustering_obj, tr_data_500, feature_names, num_clusters,topn_features=10):

    cluster_details = {}

    # get cluster centroids

    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]

    # get key features & questions for each cluster

    

    for cluster_num in range(num_clusters):

        cluster_details[cluster_num] = {}

        cluster_details[cluster_num]['cluster_num'] = cluster_num

        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]

        cluster_details[cluster_num]['key_features'] = key_features

        qnss = tr_data_500[tr_data_500['Clusters'] == cluster_num]['example_id'].values.tolist()

        cluster_details[cluster_num]['Questions'] = qnss

    return cluster_details
def print_cluster_data(cluster_data):

    # print cluster details

    for cluster_num, cluster_details in cluster_data.items():

        print('Cluster {} details:'.format(cluster_num))

        print('-'*20)

        print('Key features:', cluster_details['key_features'])

        print("Example ID's in this cluster:")

        print(', '.join(cluster_details['Questions']))

        print('='*80)
# Get clustering analysis data



cluster_data = get_cluster_data(clustering_obj=km_obj, tr_data_500=tr_data_500, feature_names=feature_names, num_clusters=num_clusters, topn_features=5)



# print clustering analysis results to see what are those features that come under the same cluster



print_cluster_data(cluster_data)
## Importing and Apply PCA



cosine_distance = 1 - cosine_similarity(feature_matrix)



from sklearn.decomposition import PCA



pca = PCA(n_components=2) # project from 784 to 2 dimensions



principalComponents = pca.fit_transform(cosine_distance)



p_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



p_df.shape
# Explaining the Variance ratio



print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# Plot the first two principal components of each point to learn about the data:



from pylab import rcParams

rcParams['figure.figsize'] = 17, 9



plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 5, c=clusters, cmap='Spectral')



plt.gca().set_aspect('equal', 'datalim')



plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))



plt.title('Visualizing the clusters', fontsize=25);



plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')