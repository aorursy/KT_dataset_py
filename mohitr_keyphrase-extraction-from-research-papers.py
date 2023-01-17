#Installing pke



!pip install git+https://github.com/boudinfl/pke.git
import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.cluster import KMeans

from sklearn.cluster import AffinityPropagation

import re

import pke
papers = pd.read_csv('../input/201812_CL_Github.csv')
papers.head()
papers.shape

#Total 106 papers given
#Keyphrase extraction(top 10) from abstracts using textrank algorithm



def extract_keyphrases(caption, n):

    extractor = pke.unsupervised.TextRank() 

    extractor.load_document(caption)

    extractor.candidate_selection()

    extractor.candidate_weighting()

    keyphrases = extractor.get_n_best(n=n, stemming=False)

    print(keyphrases,"\n")

    return(keyphrases)

    

papers['Abstract_Keyphrases'] = papers.apply(lambda row: (extract_keyphrases(row['Abstract'],10)),axis=1)
#titles & keyphrases



papers.loc[:,['Title','Abstract_Keyphrases']]
titles = papers['Title']
titles[1]
count_vectorizer = CountVectorizer()

counts = count_vectorizer.fit_transform(titles)

tfidf_vectorizer = TfidfTransformer().fit(counts)

tfidf_titles = tfidf_vectorizer.transform(counts)

tfidf_titles
#Affinity Propogation

X = tfidf_titles

clustering = AffinityPropagation().fit(X)

clustering 



content_affinity_clusters = list(clustering.labels_)

content_affinity_clusters
papers['title_cluster'] = content_affinity_clusters
#Let's check all papers in cluster 11



papers_cluster11 = papers.loc[papers['title_cluster']==11,['Title','Abstract_Keyphrases']]
papers_cluster11
dict(sorted(papers_cluster11.values.tolist())) 