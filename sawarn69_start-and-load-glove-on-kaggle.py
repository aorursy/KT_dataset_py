from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

embeddings_index = {}

with open('/kaggle/input/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embeddings_index[word]=vectors

f.close()
print('Total %s word vectors.' % len(embeddings_index))
first_20_words_in_glove=list(embeddings_index.keys())[:20]

print('first 20 tokens')

print(first_20_words_in_glove)
embeddings_index['the']
embeddings_index['earthquake']