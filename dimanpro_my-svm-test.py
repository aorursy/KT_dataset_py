import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

DEBUG = False
if (DEBUG):
    train = pd.read_json('./train.json')
    test = pd.read_json('./test.json')
else:
    train = pd.read_json('../input/train.json')
    test = pd.read_json('../input/test.json')
train.ingredients = train.ingredients.apply(lambda l: ", ".join(l))
test.ingredients = test.ingredients.apply(lambda l: ", ".join(l))
train.head()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# words - (\w+?)(?:,\s|\s|$)    ingredients - (.+?)(?:,\s|$)
vectorizerIngr = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,\
                             max_features = 50000, binary = False, token_pattern=r'(.+?)(?:,\s|$)') 
vectorizerIngr.fit(train['ingredients'])
bagOfWords = vectorizerIngr.transform(train['ingredients'])
bagOfWordsTest = vectorizerIngr.transform(test['ingredients'])

vectorizerWords = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,\
                             max_features = 50000, binary = False, token_pattern=r'(\w+?)(?:,\s|\s|$)') 
vectorizerWords.fit(train['ingredients'])
bagOfWordsWords = vectorizerWords.transform(train['ingredients'])
bagOfWordsTestWords = vectorizerWords.transform(test['ingredients'])

vectorizerTfDf =  TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,\
                             max_features = 50000, binary = False, token_pattern=r'(\w+?)(?:,\s|\s|$)')

vectorizerTfDf.fit(train['ingredients'])
bagOfWordsTfDf = vectorizerTfDf.transform(train['ingredients'])
bagOfWordsTestTfDf = vectorizerTfDf.transform(test['ingredients'])
import warnings, sklearn
import scipy.sparse as sparse
from sklearn import *
bagOfWords = bagOfWords.astype('float')
bagOfWordsWords = bagOfWordsWords.astype('float')
from sklearn.cluster import KMeans
#clusters = KMeans(n_clusters=2, random_state=0).fit_predict(data0)
#clusterDummies = pd.get_dummies(clusters, prefix='cluster')

kolvoWords = np.sum(bagOfWordsWords, axis=1)
kolvoIngr = np.sum(bagOfWords, axis=1)
lambdaKolvoIngr =  0.0606
lambdaKolvoWords = 0.05
lambdaTfDf = 1
lambdaBagOfW = 0.2121
lambdaWords = 0.116
#data1 = sparse.hstack((data0, clusterDummies*lambdaClusters, kolvo*lambdaKolvo))
data1 = sparse.hstack((lambdaTfDf * bagOfWordsTfDf, \
                       bagOfWords * lambdaBagOfW, \
                       bagOfWordsWords * lambdaWords, \
                       kolvoWords*lambdaKolvoWords, \
                       kolvoIngr*lambdaKolvoIngr))

SVM = svm.LinearSVC(C=0.3)
SVM.fit(data1, train['cuisine'])
bagOfWordsTest = bagOfWordsTest.astype('float')
bagOfWordsTestWords = bagOfWordsTestWords.astype('float')

kolvoWordsTest = np.sum(bagOfWordsTestWords, axis=1)
kolvoIngrTest = np.sum(bagOfWordsTest, axis=1)

dataTest = sparse.hstack((lambdaTfDf * bagOfWordsTestTfDf, \
                       bagOfWordsTest * lambdaBagOfW, \
                       bagOfWordsTestWords * lambdaWords, \
                       kolvoWordsTest*lambdaKolvoWords, \
                       kolvoIngrTest*lambdaKolvoIngr))
#res = SVM.predict(dataTest)"""
class FuncCreator:
    def __init__(self):
        self.dict_cuisine = dict()
        self.ind = 0
    def __call__(self, it):
        x = it.lower()
        if not (x in self.dict_cuisine.keys()):
            self.dict_cuisine[x] = self.ind
            self.ind = self.ind + 1
        return it
    
    def getNameFromId(self, id):
        for key, val in self.dict_cuisine.items():
            if (val == id):
                return key
        return "UNKNOW"
    
class MySVM:
    def __init__(self, C=0.3):
        self.SVMS = dict()
        self.rep = FuncCreator()
        self.keys=[]
        self.C=C
        
    def learn(self, data1, y, res, ar=[]):
        self.rep = FuncCreator()
        y.apply(self.rep)
        self.keys=np.array(list(self.rep.dict_cuisine.keys()))
        i = 0
        for key in self.keys:
            #print(key)
            ndata=y==key
            #print(ndata)
            if (i >= len(ar)):
                SVM = svm.LinearSVC(C=self.C)
            else:
                SVM = svm.LinearSVC(C=ar[i])
            SVM.fit(data1, ndata)
            self.SVMS[key] = SVM
            i = i + 1
            
    def predict(self, data, ar=[]):
        arr = np.zeros((data.shape[0],len(self.keys)))
        i = 0
        for key in self.keys:
            if (len(ar) == 0 or ((len(ar)) < len(self.rep.dict_cuisine.keys()))):
                arr[:,i]=self.SVMS[key].decision_function(data)
            else:
                arr[:,i]=ar[i] * self.SVMS[key].decision_function(data)
            i = i+1
        return self.keys[np.argmax(arr, axis=1)]
    
SVM = MySVM(C=0.29)
SVM.learn(data1, train['cuisine'], 0, [0.25, 0.4, 0.25, 0.25, 0.1, 0.3, 0.3, 0.3, 0.2, 0.35, 0.35, 0.5, 0.25, 0.3, 0.3, 0.35, 0.6, 0.4, 0.55, 0.45])
res = SVM.predict(dataTest)
f =open("svm_output.csv", "w")
f.write("id,cuisine\n")
i = 0
for item in test["id"]:
    f.write(str(item))
    f.write(",")
    f.write(res[i])
    f.write("\n")
    i= i +1
    
    
f.close()





