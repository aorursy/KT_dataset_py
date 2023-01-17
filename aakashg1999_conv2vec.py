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
csv_path='../input/FC-CategoryV1.xlsx'
main=pd.read_excel(csv_path, encoding = "ISO-8859-1")

main.head()
# main.loc[(main.Category== 'A')|(main.Category=='A=Power Generation')]
pd.options.display.max_colwidth = 1000
catA=main.loc[(main.Category== 'A')|(main.Category=='A=Power Generation')]
catA['Abstract - DWPI Use']
strA=pd.Series.to_string(catA['Abstract - DWPI Use'],index=False)
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
words = word_tokenize(strA)
stemmed_catA=[]
for w in words:
    stemmed_catA.append(ps.stem(w))

print(stemmed_catA)
    
import re

puncts=['(',')',',','.',';',':',',']
from nltk.corpus import stopwords
stope = stopwords.words('english')
stope.append("e.g")
stope.append("as")
stope.append("ga")
filtered_stemmed_catA= []
for c in stemmed_catA:
    if c in puncts:
        pass
    elif c in stope:
        pass
    else:
        filtered_stemmed_catA.append(c)
        
print(filtered_stemmed_catA)        
def sentence_to_wordlist(raw):
    raw=str(raw)
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words
sentences = []
for raw_sentence in filtered_stemmed_catA:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(filtered_stemmed_catA))
import gensim.models.word2vec as w2v
import sklearn.manifold
import multiprocessing
# Dimensionality of the resulting word vectors.
num_features = 300

# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()
print(num_workers)

# Context window length.
context_size = 7

# Downsample setting for frequent words.
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1
catA2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
catA2vec.build_vocab(sentences)
epochs=len(filtered_stemmed_catA)
catA2vec.train(filtered_stemmed_catA,total_examples=epochs, epochs=2)
import sklearn.manifold
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = catA2vec.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[catA2vec.wv.vocab[word].index])
            for word in catA2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
points.head()
import matplotlib.pyplot
points.plot.scatter("x", "y", s=10, figsize=(20, 12), color='red')
catA2vec.most_similar("oxygen")