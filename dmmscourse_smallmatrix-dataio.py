# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import pairwise_distances

from scipy.stats import spearmanr
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read data

simplex999 = pd.read_csv('../input/nicn/simlex999_rus_without_dupl.csv')

simplex999.head()
# Plot the distribution of original distances

sns.distplot(simplex999.distance_i_j, kde_kws={"shade":True, "color":"r", "bw":0.1});
# Split words in pairs

simplex999['word_i'] = [pair[0] for pair in simplex999.word_i_word_j.str.split('_')]

simplex999['word_j'] = [pair[1] for pair in simplex999.word_i_word_j.str.split('_')]

simplex999.head()
# Collect unique words

vocabulary = sorted({x for pair in [s.split('_') for s in simplex999.word_i_word_j] for x in pair})

wordnum = len(vocabulary)

print(wordnum)
# Enumerate the words

indexVoc = dict(zip(vocabulary,range(wordnum)))

pd.DataFrame.from_dict(indexVoc, orient='index').head()
# Index the words

simplex999['wid_i'] = [indexVoc[w] for w in simplex999.word_i]

simplex999['wid_j'] = [indexVoc[w] for w in simplex999.word_j]

simplex999.head()
# Create symmetric distance matrix

origdistmatrix = np.empty((wordnum,wordnum))

origdistmatrix[:,:] = np.NaN

origdistmatrix[simplex999.wid_i,simplex999.wid_j] = simplex999.distance_i_j

origdistmatrix[simplex999.wid_j,simplex999.wid_i] = simplex999.distance_i_j
# Metric spaces do not support non-trivial reflexion

origdistmatrix[np.arange(wordnum),np.arange(wordnum)] = 10

# Metric spaces do not support non-symmetric relations

origdistmatrix = (origdistmatrix+origdistmatrix.T)/2
# Symmetricity test

assert(np.allclose(origdistmatrix, origdistmatrix.T, equal_nan=True))
# Plot and evaluate relation

sns.scatterplot(x=simplex999.distance_i_j, y=origdistmatrix[simplex999.wid_i,simplex999.wid_j]);

spearmanr(simplex999.distance_i_j, origdistmatrix[simplex999.wid_i,simplex999.wid_j])
n = 3

# Our method to get n-dimensional representations

X = np.random.random((wordnum,n))
# Plot representations

sns.scatterplot(x=X[:,0], y=X[:,1]);
# Prepare representation for writing

result=pd.DataFrame(data=X, columns=np.arange(n)+1, index=vocabulary)

result.head()
# Write representation to csv

result.to_csv(index_label='word');

# TODO: add your own filename
# Calculate new Euclidean distances

deriveddistmatrix = pairwise_distances(X)

assert(deriveddistmatrix.shape == (wordnum,wordnum))
# Plot and evaluate relation

sns.scatterplot(x=simplex999.distance_i_j, y=deriveddistmatrix[simplex999.wid_i,simplex999.wid_j]);

spearmanr(simplex999.distance_i_j, deriveddistmatrix[simplex999.wid_i,simplex999.wid_j])