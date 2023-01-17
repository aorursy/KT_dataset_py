# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/UserRatings2.csv", index_col= 'JokeId')

print(df1.shape)

df1.head()
df2 = pd.read_csv("../input/UserRatings1.csv", index_col= 'JokeId')

print(df2.shape)

df2.head()
df = pd.merge(df2, df1, left_index = True, right_index = True)
df.head()
df.shape
from sklearn.decomposition import PCA



reducer = PCA(100)

X_reduced = reducer.fit_transform(df.fillna(0.0).values)

reducer = PCA(2)

X_reduced = reducer.fit_transform(X_reduced)
reducer.explained_variance_ratio_
import matplotlib.pyplot as plt



plt.scatter(X_reduced[:,0], X_reduced[:,1])

plt.show()
from sklearn.neighbors import NearestNeighbors
df.shape
df.head()
df.iloc[1, :].values.shape
df.iloc[1, :].values.reshape(1,-1).shape
df.iloc[1, :].values.reshape(-1,1).shape
df.shape
a = np.array([[1],[2],[3],[4],[5]]).reshape(1,-1)
a.shape
df.head()
df.transpose().head()
M = np.random.rand(1000,1000)
M.shape
M
%timeit 2 * M
print( 1 * 100 * 100)
%timeit M[0] * M 



print(100 * 100 * 100)
%timeit M * M
print(100 * 100 * 100)
df = df.transpose()
df.shape
df1 = df.iloc[:100]
df2 = df.iloc[100:200]
x = df1.iloc[1]
from sklearn.metrics.pairwise import cosine_similarity
cs1 = cosine_similarity(x.values.reshape(1,-1), df1)
cs2 = cosine_similarity(x.values.reshape(1,-1), df2)
cs1[0][1] = 0
print(cs1)
print(cs2)
print(np.argmax(cs1))

print(np.argmax(cs2))
r = [39,13]

cs = [cs1[0][39], cs2[0][13]]

print(cs)
np.argmax(cs)
r[1]