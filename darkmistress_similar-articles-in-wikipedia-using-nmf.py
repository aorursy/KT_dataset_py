#import libraries

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
df = pd.read_csv('../input/dataset101/wikipedia-vectors.csv', index_col=0)

articles = csr_matrix(df.transpose())

titles = list(df.columns)

df
#necessary import

from sklearn.decomposition import NMF

from sklearn.preprocessing import normalize



# Create an NMF model: nmf

model = NMF(n_components = 6)
# Fit the model to articles

model.fit(articles)



# Transform the articles: nmf_features

nmf_features = model.transform(articles)



# Print the NMF features

print(nmf_features)



# Normalize the NMF features: norm_features

norm_features = normalize(nmf_features)



# Create a DataFrame: df

df = pd.DataFrame(norm_features, index=titles)



# Select the row corresponding to 'Anyone': article

article = df.loc['Adam Levine']



# Compute the dot products: similarities

similarities = df.dot(article)



# Display those with the largest cosine similarity

print(similarities.nlargest())