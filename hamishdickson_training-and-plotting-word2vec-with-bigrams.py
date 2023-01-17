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
import string

import gensim

from gensim.models import phrases, word2vec

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



matplotlib.style.use('ggplot')



%matplotlib inline
df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')

df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')

df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False, delim_whitespace=True, header=0, escapechar='\\')



pd.set_option('display.max_colwidth', -1)

df_IV.columns = ['speaker','text']

df_V.columns = ['speaker', 'text']

df_VI.columns = ['speaker', 'text']
df_IV.head(4)
def prep_text(in_text):

    return in_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)

df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)

df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)

df_IV.head(5)
df = pd.concat([df_IV, df_V, df_VI])



sentences = df.clean_text.values

# for idx, row in df.iterrows():

#     sentences.append(row['clean_text'])



df.head(5)
bigrams = phrases.Phrases(sentences)
print(bigrams["this is the death star".split()])
bigrams[sentences]



model = word2vec.Word2Vec(bigrams[sentences], size=50, min_count=3, iter=20)
model.wv.most_similar('death_star')
vocab = list(model.wv.vocab)

len(vocab)
X = model[vocab]
pca = PCA(n_components=3, random_state=11, whiten=True)

clf = pca.fit_transform(X)



tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y', 'z'])



tmp.head(3)
tmp = tmp.sample(150)
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(tmp['x'], tmp['y'], tmp['z'], alpha = 0.5)



for word, row in tmp.iterrows():

    x, y, z = row

    pos = (x, y, z)

    ax.text(x, y, z, s=word, size=8, zorder=1, color='k')

    

plt.title('w2v map - PCA')

plt.show()
tsne = TSNE(n_components=3, random_state=11)

clf = tsne.fit_transform(X)



tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y', 'z'])



tmp.head(3)
tmp = tmp.sample(150)
fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(tmp['x'], tmp['y'], tmp['z'], alpha = 0.5)



for word, row in tmp.iterrows():

    x, y, z = row

    pos = (x, y, z)

    ax.text(x, y, z, s=word, size=8, zorder=1, color='k')

    

plt.title('w2v map - t-SNE')

plt.show()