# INSTALLATION:



!pip install pyldavis
from collections import defaultdict, Counter

from configparser import ConfigParser, ExtendedInterpolation

from IPython.core.display import display, HTML

from IPython.display import Image

from IPython.lib.display import YouTubeVideo

from gensim import corpora, models

from gensim.models.ldamodel import LdaModel

from gensim.models.callbacks import DiffMetric

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pyLDAvis  # conda install pyldavis

import pyLDAvis.gensim

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

import spacy

from spacy.matcher import Matcher, PhraseMatcher

from spacy.lang.en.stop_words import STOP_WORDS



%matplotlib inline
# PCA

# SOURCE: https://intoli.com/blog/pca-and-svd/

# Image("https://s3.amazonaws.com/nlp.practicum/pca.png", width=700)
# create an example dataframe

X_df = pd.DataFrame([[-1, 1], [-2, 1], [-3, 2], [1, -1], [2, -1], [3, -2]])

X_df
# plot the data

X_df.plot.scatter(x=0, y=1)
# import PCA

from sklearn.decomposition import PCA



# run PCA to create 2 principal components

pca = PCA(n_components=2)

fit_pca = pca.fit_transform(X_df)



# create the transformed matrix

pca_df = pd.DataFrame(fit_pca, columns=['component 1','component 2'])

pca_df
# plot the prinipal components

pca_df.plot.scatter(x='component 1', y='component 2')



# increase range of y-axis to match the original visualization

plt.ylim(-2,2)
print('Singular values: {}'.format(pca.singular_values_.round(2)))



# Singular Values are the l2 norm of each component

# e.g. np.sqrt(sum([i**2 for i in pca_df['component 1']]))

print('Singluar Value 1: {}'.format(np.linalg.norm(pca_df['component 1'], ord=2).round(2)))

print('Singluar Value 2: {}'.format(np.linalg.norm(pca_df['component 2'], ord=2).round(2)))



print('\nExplained variance ratio: {}'.format(pca.explained_variance_ratio_))
# view the first prinicipal component

pca_df['component 1']
# square all values

pca_df['component 1'] ** 2
# take the sum of squares

sum(pca_df['component 1'] ** 2)
# take the sqaure root of the sum of squares

np.sqrt(sum(pca_df['component 1'] ** 2))
print('Components:\n {}'.format(pca.components_.round(2)))
# recreate PCA using SVD (we will explain this calculation in the next section)

u,s,vt = np.linalg.svd(X_df, full_matrices=False)



print("singular values =\n {} \n".format(np.round(s, 2)))

print("components =\n {} \n".format(np.round(vt, 2)))



# pca

pd.DataFrame(u*s)
# SVD

# Image("https://s3.amazonaws.com/nlp.practicum/svd_graph.png", width=400, height=200)
# SVD

# Image("https://s3.amazonaws.com/nlp.practicum/svd3.png", width=400)
# reduced SVD

# Image("https://s3.amazonaws.com/nlp.practicum/reduced_svd.png", width=500, height=500)
# SVD

# Image("https://s3.amazonaws.com/nlp.practicum/svd2.png", width=700)
# Truncated SVD

# SOURCE: https://www.researchgate.net/figure/Singular-value-decomposition-SVD-and-a-truncated-SVD-matrix_fig1_227060873

# Image("https://s3.amazonaws.com/nlp.practicum/truncated_svd.png", width=500)
# SVD

Image("https://s3.amazonaws.com/nlp.practicum/svd4.png", width=700)
# reduced SVD equation

Image("https://s3.amazonaws.com/nlp.practicum/svd_truncated_equation.png", 

      width=500, height=200)
# SVD vs PCA

Image("https://s3.amazonaws.com/nlp.practicum/svd_vs_pca.png", width=1000)
# SVD vs PCA

# Image("https://s3.amazonaws.com/nlp.practicum/svd_vs_pca2.png", width=700)
from sklearn.feature_extraction.text import CountVectorizer
# GENSIM_DICTIONARY_PATH = r'gensim_dictionary_path.txt' 

# GENSIM_CORPUS_PATH = r'gensim_corpus_path.txt'



# CLEANED_TEXT_PATH = r'https://raw.githubusercontent.com/Alexjmsherman/nlp_practicum_cohort3_student/master/raw_data/cleaned_text/cleaned_text.txt?token=ABXRUPVMSIR3QOBGSWUI5SS5CA6AY'

CLEANED_PATH = "../input/datafornlp/text.csv"

texts = pd.read_csv(CLEANED_PATH, sep='\t',header=None)

texts = [line[0].split() for line in texts.values]

print(texts[0])
# combine tokens from the first few lists into sentences

svd_data = [' '.join(text) for text in texts[0:8]]



# create a document term matrix of the token counts

vect = CountVectorizer(max_features=10, stop_words='english')

dtm = vect.fit_transform(svd_data)



# create a dataframe

vocab = vect.get_feature_names()

df = pd.DataFrame(dtm.toarray(), columns=vocab)

df
# decompose the matrix using SVD

U, s, VT = np.linalg.svd(df, full_matrices=False)

S = np.diag(s)
# what are U, S and V

print("U =\n", np.round(U, decimals=2), "\n")

print("S =\n", np.round(S, decimals=2), "\n")

print("V^T =\n", np.round(VT, decimals=2), "\n")
# U is othonormal

# These vectors are orthogonal to one another; form a basis for the reduced space



# each vector is normalized (unit vector)

# multiply by itself returns 1

col1 = np.array([i[0] for i in U])

print(col1, '\n')

print('vector 1: {}'.format(round(col1.dot(col1), 2), '\n'))



col2 = np.array([i[1] for i in U])

print('vector 2: {}'.format(round(col2.dot(col2), 2)))



# and each vector is orthogonal to the other vectors

# multiply different vectors returns 0

print('dot product: {}'.format(round(col1.dot(col2), 2)))
# rebuild the original matrix from U,S, and V^T

A2 = np.dot(U, np.dot(S, VT))

print("A2 =\n", A2.round(2))
# example of np.zero_like

np.zeros_like(S)
# S_reduced is the same as S but with only the top n elements kept

S_reduced = np.zeros_like(S)



# only keep top few eigenvalues

eigen_num = 3

S_reduced[:eigen_num, :eigen_num] = S[:eigen_num,:eigen_num]



# show S_rediced which has less info than original S

print("S_reduced =\n", S_reduced.round(2))
# reduce VT by S_reduced

S_reduced_VT = np.dot(S_reduced, VT)

print("S_reduced_VT = \n", S_reduced_VT.round(2))
# each Singular Value vector is a linear combination of original words

U_S_reduced = np.dot(U, S_reduced)

df = pd.DataFrame(U_S_reduced.round(2))



# show colour coded so it is easier to see significant word contributions to a topic

df.style.background_gradient(cmap=plt.get_cmap('Blues'))
# recreate using sklearn (explained below)

from sklearn.decomposition import TruncatedSVD

tsvd = TruncatedSVD(n_components=3)

tsvd.fit_transform(df).round(2)
from sklearn.decomposition import TruncatedSVD
# review data

svd_data[0]
# vectorize the text with TFIDF

vect = TfidfVectorizer(max_features=10)

fit_vect = vect.fit_transform(svd_data)

pd.DataFrame(fit_vect.toarray(), columns=vect.get_feature_names())
# retain one component

tsvd = TruncatedSVD(n_components=1)

tsvd.fit_transform(fit_vect)
# retain two components

tsvd = TruncatedSVD(n_components=2)

tsvd.fit_transform(fit_vect)