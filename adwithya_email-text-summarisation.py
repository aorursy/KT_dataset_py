import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import re

from nltk.corpus import stopwords

stopWords = stopwords.words('english')

from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
#Helper functions



def clean(email):

    return re.sub(r"^.*:\s.*|^-.*\n?.*-$|\n|^>", '', email, 0, re.MULTILINE)

def removeStopwords(sentence):

    newSentence = " ".join([i for i in sentence if i not in stopWords])

    return newSentence

def prettySentences(sentence):

    for s in sentence:

        print(s)

        print()
chunk = pd.read_csv('../input/enron-email-dataset/emails.csv', chunksize=2000)

data = next(chunk)
print(data.message[9])
myData = []

for sentence in data.message:

    myData.append(clean(sentence))
print(myData[9])
from nltk.tokenize import sent_tokenize



sentences = []



for sentence in myData:

    sentences.append(sent_tokenize(sentence))
prettySentences(sentences[9])
wordEmbeddings = {}



with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt', encoding='utf-8') as f:

    for line in f:

        values = line.split()

        key = values[0]

        wordEmbeddings[key] = np.asarray(values[1:], dtype='float32')
len(wordEmbeddings)
cleanSentences = []



# make alphabets lowercase

for email in sentences:

        email = [re.sub(r"[^a-zA-Z]", " ", s, 0, re.MULTILINE) for s in email]

        email = [re.sub(r"\s+", " ", s, 0, re.MULTILINE) for s in email]

        cleanSentences.append([s.lower() for s in email])
prettySentences(cleanSentences[9])
for i in range(len(cleanSentences)):

    cleanSentences[i] = [removeStopwords(r.split()) for r in cleanSentences[i]]
prettySentences(cleanSentences[9])
sentenceVectors = []



for email in cleanSentences:

    temp = []

    for s in email:

        if len(s) != 0:

            v = sum([ wordEmbeddings.get(w, np.zeros((100,))) for w in s.split()])/(len(s.split()) + 0.001)

        else:

            v = np.zeros((100,))

        temp.append(v)

    sentenceVectors.append(temp)
sentenceVectors[9][0]
similarityMatrix = []

for i in range(len(cleanSentences)):

    email = cleanSentences[i]

    temp = np.zeros((len(email), len(email)))

    j_range = temp.shape[0]

    k_range = temp.shape[1]

    for j in range(j_range):

        for k in range(k_range):

            if j != k:

                temp[j][k] = cosine_similarity(sentenceVectors[i][j].reshape(1, 100),

                                                             sentenceVectors[i][k].reshape(1, 100))[0][0]

    similarityMatrix.append(temp)
similarityMatrix[9]
scores = []

for i in similarityMatrix:

    nxGraph= nx.from_numpy_array(i)

    scores.append(nx.pagerank_numpy(nxGraph))

scores[9]
rankedSentences = []



for i in range(len(scores)):

    rankedSentences.append(sorted(((scores[i][j],s) for j,s in enumerate(sentences[i])), reverse=True))
myData[108]
example = 108



print(rankedSentences[example][0][1])

print(rankedSentences[example][1][1])
for i in range(len(rankedSentences)):

    print("--------------------------------")            

    if len(rankedSentences[i]) > 2:

        print(rankedSentences[i][0][1])

        print(rankedSentences[i][1][1])

    else:

        for j in range(len(rankedSentences[i])):

             print(rankedSentences[i][j][1])

    print("--------------------------------")            

    print()