# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from scipy import optimize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Read CSV
df = pd.read_csv("../input/countries of the world.csv", decimal = ',').fillna(0)

# Region encoding
le = preprocessing.LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

# Scaling data
scaler = MinMaxScaler()
df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])

# Prepare Dataset
X = df[df.columns[2:]].values
y = df[df.columns[1]].values
# t-SNE
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X)
# Plot
import matplotlib.pyplot as plt
palette = ['red', 'green', 'blue', 'gray', 'violet', 'black', 'yellow', 'brown', 'coral', 'darkgreen', 'cyan']
colors = list(map(lambda x: palette[x], y))
fig = plt.figure(figsize=(10, 10)) 
ax = fig.subplots()
ax.scatter(X_embedded[:,0], X_embedded[:,1], c = colors)

for i, text in enumerate(df['Country']):
    ax.annotate(text, (X_embedded[i,0], X_embedded[i,1]))
    
plt.show()
