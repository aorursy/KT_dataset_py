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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

%matplotlib inline

df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False)
df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False)
df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False)
pd.set_option('display.max_colwidth', -1)
df_IV.columns = ['text']
df_V.columns = ['text']
df_VI.columns = ['text']

df_IV.head(5)
def prep_text(in_text):
    return in_text.split('"')[3:-1][0].lower().translate(str.maketrans("", "", string.punctuation)).split()
df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)
df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)
df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)
df_IV.head(5)
sentences_iv = df_IV['clean_text']
sentences_v = df_V['clean_text']
sentences_vi = df_VI['clean_text']

# use a min count of 3 since our corpus is quite small
model = gensim.models.Word2Vec(min_count=3, window=5, iter=50)
model.build_vocab(sentences_iv)
model.train(sentences_iv, total_examples=model.corpus_count, epochs=model.epochs)
model.wv.most_similar('ship')
model.wv.most_similar(negative=['ship'])
characters = ['kenobi', 'han', 'jabba', 'leia', 'luke', 'threepio', 'r2', 'vader', 'wedge', 'yoda', 'chewbacca', 'lando']
vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
vocab
X = model[vocab]
cluster_num = 3

kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X)
cluster = kmeans.predict(X)
pca = PCA(n_components=2, random_state=11, whiten=True)
clf = pca.fit_transform(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])

tmp.head(3)
tmp['cluster'] = None
tmp['c'] = None

count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title('Episode IV')
plt.show()
model.build_vocab(sentences_v, update=True)
model.train(sentences_v, total_examples=model.corpus_count, epochs=model.epochs)

vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
    
X = model[vocab]
clf = pca.transform(X)
cluster = kmeans.predict(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])

tmp['cluster'] = None
tmp['c'] = None

count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title("Episodes IV and V")
plt.show()
model.build_vocab(sentences_vi, update=True)
model.train(sentences_vi, total_examples=model.corpus_count, epochs=model.epochs)

vocab = list(model.wv.vocab)
vocab = list(filter(lambda x: x in characters, vocab))
    
X = model[vocab]

clf = pca.transform(X)
cluster = kmeans.predict(X)

tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y'])
    
tmp['cluster'] = None
tmp['c'] = None
count = 0
for index, row in tmp.iterrows():
    tmp['cluster'][index] = cluster[count]
    tmp['c'][index] = characters[count]
    count += 1
    
for i in range(cluster_num):
    values = tmp[tmp['cluster'] == i]
    plt.scatter(values['x'], values['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, cat, character = row
    pos = (x, y)
    plt.annotate(character, pos)
    
plt.axis('off')
plt.title("Episodes IV, V and VI")
plt.show()
