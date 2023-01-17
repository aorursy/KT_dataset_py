#Importing the required libraries

import re

import math

import numpy as np

import pandas as pd

import string

import scipy as sp

import nltk

import time

import operator

from scipy import *

from scipy.sparse import *

from collections import defaultdict

from collections import Counter

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer 

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import os

print(os.listdir("../input"))
#remove words less than length of 4

def filterLen(docs, minlen):

    return [ [t for t in d if len(t) >= minlen ] for d in docs ]
#Building Bag of words

def word_bag(docs):

#     nrows = len(docs)

    idx = {}

    tid = 0

    for d in docs:

        for w in d:

            if w not in idx:

                idx[w] = tid

                tid += 1

    return idx

    

#Building the sparse matrix

from collections import Counter

from scipy.sparse import csr_matrix

def build_matrix(docs, idx):

    r""" Build sparse matrix from a list of documents, 

    each of which is a list of word/terms in the document.  

    """

    nrows = len(docs)

    nnz = 0

    for d in docs:

        nnz += len(set(d))

        

    ncols = len(idx)

        

    # set up memory

    ind = np.zeros(nnz, dtype=np.int)

    val = np.zeros(nnz, dtype=np.double)

    ptr = np.zeros(nrows+1, dtype=np.int)

    i = 0  # document ID / row counter

    n = 0  # non-zero counter

    # transfer values

    for d in docs:

        cnt = Counter(d)

        keys = list(k for k,_ in cnt.most_common())

        l = len(keys)

        for j,k in enumerate(keys):

            #print(keys)

            if(k in idx):

                

                ind[j+n] = idx[k]

                val[j+n] = cnt[k]

        ptr[i+1] = ptr[i] + l

        n += l

        i += 1

            

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)

    mat.sort_indices()

    

    return mat
# scale matrix and normalize its rows

def csr_idf(mat, copy=False, **kargs):

    r""" Scale a CSR matrix by idf. 

    Returns scaling factors as dict. If copy is True, 

    returns scaled matrix and scaling factors.

    """

    if copy is True:

        mat = mat.copy()

    nrows = mat.shape[0]

    nnz = mat.nnz

    ind, val, ptr = mat.indices, mat.data, mat.indptr

    # document frequency

    df = defaultdict(int)

    for i in ind:

        df[i] += 1

    # inverse document frequency

    for k,v in df.items():

        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory

    # scale by idf

    for i in range(0, nnz):

        val[i] *= df[ind[i]]

        

    return df if copy is False else mat



def csr_l2normalize(mat, copy=False, **kargs):

    if copy is True:

        mat = mat.copy()

    nrows = mat.shape[0]

    nnz = mat.nnz

    ind, val, ptr = mat.indices, mat.data, mat.indptr

    # normalize

    for i in range(nrows):

        rsum = 0.0    

        for j in range(ptr[i], ptr[i+1]):

            rsum += val[j]**2

        if rsum == 0.0:

            continue  # do not normalize empty rows

        rsum = 1.0/np.sqrt(rsum)

        for j in range(ptr[i], ptr[i+1]):

            val[j] *= rsum

            

    if copy is True:

        return mat
#kNN Function

class knn_main():

    def __init__(self):

        pass

    

    def fit(self, x_train, y_train):

        self.x_train = x_train

        self.y_train = y_train

    

    def predict(self, x_test,k,eps):

        self.x_test = x_test

        y_predict = []

        print(x_test.shape[0])

        for i,test_case in enumerate(x_test):

            temp = self.compute_distances(test_case, self.x_train)

            c = temp.tocoo(copy=True)

            d = self.sort_coo(c)

       #print(d[0])

            t = d[0:k]

        #print(t)

            dict_l ={}

        #print(k)

            for z,x in enumerate(t):

                index = x[1]

                similarity = x[2]

                label_t = self.y_train[index]

                if(z>0):

                    

                    if similarity > eps:

                        if label_t not in dict_l:

                            dict_l[label_t]=1

                        else:

                            dict_l[label_t]+=1

                else:

                    dict_l[label_t] =1

                

            m = max(dict_l.items(),key=operator.itemgetter(1))[0]

            y_predict.append(m)

            #print("test case:", i+1,"Predicted :", m)

        return y_predict

    def sort_coo(self,m):

        tuples = zip(m.row, m.col, m.data)

        return sorted(tuples, key=lambda x: (x[2]),reverse=True)

    

    def compute_distances(self, X_test,train):        

        dot_pro = np.dot(X_test, train.T)

        return(dot_pro)
#Seperation of labels and text data

labels = []

texts = []

with open("../input/train.dat", "r") as fh:

    train_lines = fh.readlines() 

for line in train_lines:

    splitline = line.split('\t')

    labels.append(splitline[0])

#     ps = word_tokenize()

    texts.append(nltk.word_tokenize(splitline[1].lower()))



len(texts)

# docs1 = filterLen(texts, 4)

# docs2 = filterLen(tex,4)
train_text =  filterLen(texts, 4)

wordbag = word_bag(train_text)

train_mat = build_matrix(train_text,wordbag)

mat2 = csr_idf(train_mat, copy=True)

norm_train_mat = csr_l2normalize(mat2, copy=True)
X_train, X_test, y_train,y_test = train_test_split(texts,labels,test_size = 0.3)

test_mat = build_matrix(X_test,wordbag)

mat3 = csr_idf(test_mat, copy=True)

norm_test_mat = csr_l2normalize(mat3, copy=True)
classifier = knn_main()

classifier.fit(norm_train_mat,labels)

g = classifier.predict(norm_test_mat,100,0.1)
cm=confusion_matrix(y_test ,g)

print(cm)

print(classification_report(y_test, g))
tst_tex = []

with open("../input/test.dat", "r") as fr:

    test_lines = fr.readlines() 

for line in test_lines:

#         splitline = line.split()

    tst_tex.append(nltk.word_tokenize(line.lower()))

test_text = filterLen(tst_tex, 4)
test_mat = build_matrix(test_text,wordbag)

mat3 = csr_idf(test_mat, copy=True)

norm_test_mat = csr_l2normalize(mat3, copy=True)

# norm_test_mat
classifier = knn_main()

classifier.fit(norm_train_mat,labels)

#print(mat5)
g = classifier.predict(norm_test_mat,100,0.1)
with open('program.dat', 'w') as f:

        for cls in g:

            f.write("%s\n" % cls)