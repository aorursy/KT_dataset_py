# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
f = open('../input/receptxt/text.txt', encoding='UTF-8')

all_recepts = f.read()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import one_hot

from keras.preprocessing.text import text_to_word_sequence

# docs = ['Well done!',

# 		'Good work',

# 		'Great effort',

# 		'nice work',

# 		'Excellent!']

# # create the tokenizer

# t = Tokenizer()

# # fit the tokenizer on the documents

# t.fit_on_texts(all_recepts)

# # summarize what was learned

# print(t.word_counts)

# print(t.document_count)

# print(t.word_index)

# print(t.word_docs)

# # integer encode documents

# encoded_docs = t.texts_to_matrix(all_recepts, mode='tfidf')

# print(encoded_docs[:10])
import gzip

import gensim 

import logging

from gensim.test.utils import common_texts, get_tmpfile

# seq_text = text_to_word_sequence(all_recepts)



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

 
class MySentences(object):

    def __init__(self, dirname):

        self.dirname = dirname

 

    def __iter__(self):

        for fname in os.listdir(self.dirname):

            for line in open(os.path.join(self.dirname, fname)):

                yield line.split()
MySentences('../input/receptxt')


def read_input(input_file):

    """This method reads the input file which is in gzip format"""

 

    logging.info("reading file {0}...this may take a while".format(input_file))

    with open(input_file, 'r',  encoding='UTF-8') as f:

        for i, line in enumerate(f): 

            if (i % 10000 == 0):

                logging.info("read {0} reviews".format(i))

            # do some pre-processing and return list of words for each review

            # text

            yield gensim.utils.simple_preprocess(line)
documents = read_input('../input/receptxt/text.txt',)
path = get_tmpfile("word2vecRecept.model")



model = gensim.models.Word2Vec(MySentences('../input/receptxt'),size=150,window=10,min_count=2,workers=10,iter=10)



model.save("/kaggle/working/word2vecRecept.model")
model.save("/kaggle/working/word2vecReceptNORM.model")
word_vectors = model.wv
from gensim.models import KeyedVectors

path = get_tmpfile("/kaggle/working/wordvectors.kv")

model.wv.save(path)

wv = KeyedVectors.load("/kaggle/working/model.wv", mmap='r')

# vector = wv['computer']  # numpy vector of a word
model.wv.most_similar(positive='помыть')
method = ['варить', 'жарить', 'парить', 'тушить', 'помыть']

blu = ['гороховый', 'пюре', 'яичница','ризотто']

items = method + blu



item_vectors = [(item, model[item])

                   for item in items

                    if item in model

               ]
import numpy

import umap

import matplotlib.pyplot as plt

import sys

import random
vectors = np.asarray([x[1] for x in item_vectors])

lengths = np.linalg.norm(vectors, axis=1)

norm_vectors = (vectors.T/lengths).T



um = umap.UMAP(n_neighbors=norm_vectors.shape[0] - 1, n_components=2, random_state=42)
u = um.fit_transform(norm_vectors)
x = u[:,0]

y = u[:,1]

fig, ax = plt.subplots()

ax.scatter(x,y)

for item, x1,y1 in zip(item_vectors, x,y):

    ax.annotate(item[0],(x1+.05,y1))

plt.show()
positive = ['варить', 'жарить', 'парить', 'тушить', 'помыть']
negative = random.sample(model.wv.vocab.keys(),5000)

negative[:5]
labelled = [(p,1) for p in positive] + [(n,0) for n in negative]
X = np.asarray([model[w] for w,l in labelled])

y = np.asarray([l for w,l in labelled])
from sklearn.svm import SVC

TF = 0.7

cut_off = int(TF*len(labelled))

clf = SVC(kernel='linear')

clf.fit(X[:cut_off], y[:cut_off])
res = clf.predict(X[cut_off:])
res, y[cut_off:]
from sklearn.metrics import confusion_matrix

confusion_matrix(y[cut_off:], res)
!pip install umap-learn