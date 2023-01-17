%matplotlib inline

import os

import gc

import re

import sqlite3

import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

import scipy

import random



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.preprocessing import StandardScaler



from sklearn.decomposition import TruncatedSVD #can efficiently work with sparce matrices

from sklearn.decomposition import PCA #can only work with arrays



from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer
clean_data = pd.read_csv("../input/clean-amazon-fine-food-review/clean_AmazonFFR.csv")
clean_data.dropna(axis=0, how='any',inplace=True)

clean_data.reset_index(drop=True,inplace=True)

clean_data['review_type'].value_counts(normalize = True)
positive = clean_data[clean_data['review_type'] =='positive'][:20000:10].reset_index(drop=True)

negative = clean_data[clean_data['review_type'] =='negative'][:20000:10].reset_index(drop=True)



sample_data = pd.concat([positive, negative], axis=0)

print(f"{positive.shape} : {negative.shape} {sample_data.shape}")

gc.collect()
def plotReview(features, classes, transformer, title):

    

    transformed_x = transformer.fit_transform(features)

    classes = np.array(classes)

    if classes.ndim == 1:

        classes = classes.reshape(-1, 1)

    

    data = np.hstack([transformed_x, classes])

    data = pd.DataFrame(data, columns = ['dim1', 'dim2', 'review_type'])

    sns.FacetGrid(data, hue='review_type', size=4).map(plt.scatter, 'dim1', 'dim2').add_legend()

    plt.title(title)

    plt.show()
transformers = {

    'pca':PCA(n_components=2),

    'sparse_pca':TruncatedSVD(n_components=2, random_state=123)

}
count_vec = CountVectorizer(ngram_range=(1,1))

bow = count_vec.fit_transform(sample_data['Text'].values)

f"{type(bow)}, unique words:{bow.shape[1]}"
plotReview(bow.toarray(), sample_data['review_type'],transformers['pca'], 'PCA - BOW')

plotReview(bow, sample_data['review_type'],transformers['sparse_pca'], 'Sparse_PCA - BOW')
tfidf_vec = TfidfVectorizer()

tfidf_features = tfidf_vec.fit_transform(sample_data['Text'].values)



plotReview(tfidf_features.toarray(), sample_data['review_type'],transformers['pca'], 'PCA - TF_IDF')

plotReview(tfidf_features, sample_data['review_type'],transformers['sparse_pca'], 'Sparse_PCA - TF_IDF')