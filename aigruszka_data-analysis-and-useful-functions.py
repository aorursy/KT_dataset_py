import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import io
train = pd.read_csv("../input/train.csv")
train.head()
#print((train['target'] == 1).sum())
#print((train['target'] == 0).sum())

x = np.arange(2)
values = [(train['target'] == 0).sum(), (train['target'] == 1).sum()]
fig, ax = plt.subplots()
plt.bar(x, values)
plt.xticks(x, ('0', '1'))
plt.title('target column in train.csv')
plt.show()
pd.set_option('display.max_colwidth', -1)
train[train['target'] == 1].sample(10)
pd.set_option('display.max_colwidth', -1)
train[train['target'] == 0].sample(10)
from gensim.models.keyedvectors import KeyedVectors  #import this to read binary file, isn't it against rule "No custom packages" ?

def loadGoogleModel(pathToFile):
    googleModel = KeyedVectors.load_word2vec_format(pathToFile, binary=True)
    return googleModel

pathToGoogleFile = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

# googleModel = loadGoogleModel(pathToGoogleFile)
# result = googleModel.most_similar(positive=['dog'], topn=5) #you can also put the negative words ex. negative=['cat'], topn - number of top examples in return
# print(result)

def loadGloveModel(pathToFile):
    print("Loading Glove Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToGloveFile = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

# gloveModel = loadGloveModel(pathToGloveFile)
# print(gloveModel['frog'])
def loadParagramModel(pathToFile):
    print("Loading Paragram Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToParagramFile = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"

# paragramModel = loadParagramModel(pathToParagramFile)
# print(paragramModel['frog'])
def loadWikiModel(pathToFile):
    print("Loading Wiki Model")
    f = open(pathToFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done, words loaded!")
    return model

pathToWikiFile = "../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

# wikiModel = loadWikiModel(pathToWikiFile)  
# print(wikiModel['frog'])
