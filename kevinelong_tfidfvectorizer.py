
cleaned_corpus = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document',
]
corpust_term_freq  = {}
for document in cleaned_corpus:
    words = document.split(" ")
    for w in words:
        if w in corpust_term_freq:
            corpust_term_freq[w] += 1
        else:
            corpust_term_freq[w] = 1
print(corpust_term_freq)
print(len(corpust_term_freq))

document_frequency_list = []
for document in cleaned_corpus:
    words = document.split(" ")
    doc_freq = {}
    for w in words:
        if w in doc_freq:
            doc_freq[w] += 1
        else:
            doc_freq[w] = 1
    document_frequency_list.append(doc_freq)
    
print(document_frequency_list)

for i in range(len(df_list)):
    doc = df_list[i]
    keys = list(output.keys())
    for j in range(len(keys)):
        w = keys[j]
        if w in doc:
            v = doc[w] / output[w]
            print(f"({i} {j}) {v} {w}")        
        else:
            print(f"({i} {j}) 0  {w}")
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
print(X.shape)
# (4, 9)
print(X)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from IPython.display import display

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'
corpus = [documentA, documentB]
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

print('----------- compare word count -------------------')
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

series_A = pd.Series(numOfWordsA)
series_B = pd.Series(numOfWordsB)
df = pd.concat([series_A, series_B], axis=1).T
df = df.reindex(sorted(df.columns), axis=1)
display(df)

tf_df = df.divide(df.sum(1),axis='index')

n_d = 1+ tf_df.shape[0]
df_d_t = 1 + (tf_df.values>0).sum(0)
idf = np.log(n_d/df_d_t) + 1

pd.DataFrame(df.values * idf,
                  columns=df.columns )
tfidf = TfidfVectorizer(token_pattern='(?u)\\b\\w\\w*\\b', norm=None)
t = tfidf.fit_transform(corpus)
print(t)
d = t.todense()
print(d)
fn = tfidf.get_feature_names()
print(fn)
pd.DataFrame(d, columns=fn )