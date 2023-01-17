# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter

import nltk

from nltk import word_tokenize

import re



from sklearn.model_selection import train_test_split as tts

from gensim.models import doc2vec

import gensim

import json



import sys



STOP_WORDS = nltk.corpus.stopwords.words()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def clean_sentence(sent):

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', sent).lower()

    sentence = sentence.split(" ")



    for word in list(sentence):

        if word in STOP_WORDS:

            sentence.remove(word)



    sentence = " ".join(sentence)

    return sentence
import math

import numpy as np



def cosine(v1, v2):

    """

            v1 and v2 are two vectors (can be list of numbers) of the same dimensions. Function returns the cosine distance between those

            which is the ratio of the dot product of the vectors over their RS.

    """

    v1 = np.array(v1)

    v2 = np.array(v2)



    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
data = pd.read_csv('../input/train.csv')

data = data.dropna(how="any")
def concatenate(data):

    X_set1 = data['question1']

    X_set2 = data['question2']

#    y = data['is_duplicate']

    X = X_set1.append(X_set2, ignore_index=True)

    

    return X
print('Cleaning data, this might take long')

for col in ['question1', 'question2']:

    data[col] = data[col].apply(clean_sentence)
print('Splitting data to train and test sets.')

y = data['is_duplicate']

X_train, X_test, y_train, y_test = tts(data[['id','question1', 'question2']], y, test_size=0.3)



X_train.head()
import multiprocessing

cores = multiprocessing.cpu_count()

assert gensim.models.doc2vec.FAST_VERSION > -1
from gensim.models.doc2vec import Doc2Vec

from gensim.models import doc2vec

class LabeledLineSentence(object):



    def __init__(self, doc_list, labels_list):

        self.labels_list = labels_list

        self.doc_list = doc_list



    def __iter__(self):

        for idx, doc in enumerate(self.doc_list):

            yield doc2vec.TaggedDocument(words=word_tokenize(doc),

                                         tags=[self.labels_list[idx]])
X = concatenate(X_train)

labels = []

for label in X_train['id'].tolist():

    labels.append('SENT_%s_1' % label)

for label in X_train['id'].tolist():

    labels.append('SENT_%s_2' % label)



docs = LabeledLineSentence(X.tolist(), labels)

it = docs.__iter__()

model1 = Doc2Vec(it, size=12, window=8, min_count=5, workers=4)
for epoch in range(10):

    model1.train(it)

    model1.alpha -= 0.0002  # decrease the learning rate

    model1.min_alpha = model1.alpha  # fix the learning rate, no deca

    model1.train(it)
X_test.index = np.arange(0, X_test['question1'].shape[0])

y_test.index = np.arange(0, X_test['question1'].shape[0])

#print(X_test)

count = 0

for i in range(X_test['question1'].shape[0]):

    doc1 = word_tokenize(X_test['question1'][i])

    doc2 = word_tokenize(X_test['question2'][i])



    docvec1 = model1.infer_vector(doc1)

    docvec2 = model1.infer_vector(doc2)



    #print(docvec1)

    #print(docvec2)



    print(cosine(docvec1, docvec2), y_test[i])

    if count>100:

        break

    count+=1