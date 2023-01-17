import numpy as np  
import re  
import nltk  
from sklearn import metrics
from nltk.corpus import stopwords  
import pandas as pd
import csv 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from yellowbrick.text import TSNEVisualizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch



stemmer = SnowballStemmer('english')
lemmer=WordNetLemmatizer()

df = pd.read_csv("../input/Productnames.csv", sep=',',header=None,lineterminator='\n')
documents = []
for sen in df.iterrows(): 
    # Remove all the special characters 
    document = re.sub(r'\W', ' ', str(sen))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', '', document)
    #remove numbers
    document = re.sub(r'[0-9]+', '', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', '', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Stemming and Lemmatization
    document = document.split()
    document=[stemmer.stem(word) for word in document]
    document = [lemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    document = re.sub(r'name dtype object', '', document)
    document = re.sub(r'\W*\b\w{1,2}\b', '', document)

    documents.append(document)
vectorizer = TfidfVectorizer(max_features=2000,
                         		stop_words='english',
                                 use_idf=True)#, ngram_range = (2,3))
X = vectorizer.fit_transform(documents)
true_k = 50	
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000)

cluster_label=model.fit_predict(X)
Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Kmeans.csv')
score_kmeans = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_kmeans)
tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()
model = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=200, preference=None, verbose=False)

cluster_label=model.fit_predict(X)
Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Affinity.csv')
score_affinity = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_affinity)
tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()
model = SpectralClustering(n_clusters=true_k, assign_labels="discretize",random_state=0)
cluster_label=model.fit_predict(X)
Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Spectral.csv')
score_spectral = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_spectral)
tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()
model = Birch(branching_factor=100, n_clusters=true_k, threshold=0.5,compute_labels=True)
cluster_label=model.fit_predict(X)
Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_birch.csv')
score_birch = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_birch)
tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()
print('For comparison ')
print(score_kmeans)
print(score_affinity)
print(score_spectral)
print(score_birch)