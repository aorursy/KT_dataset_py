import re
import bz2
import pickle
import _pickle as cPickle

import pickle
from sys import path
from os.path import join

import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def loadAndProcessData(filePath):
    data = pd.read_csv(filePath, usecols=['comment', 'rating'])
    data = data.fillna('')
    data = data.sample(frac=1).reset_index(drop=True)

    tempList = list()
    for d in data['comment']:
        d = d.strip().lower()
        #d = re.sub('[\W_]+', '', d, flags = re.UNICODE)
        tempList.append(d)
    data['comment'] = tempList.copy()

    tempList = list()
    for d in data['comment']:
        if d.islower():
            tempList.append(d)
        else:
            tempList.append('')
    data['comment'] = tempList.copy()

    tempList = list()
    for r in data['rating']:
        r = float(r)
        tempList.append(r)
    data['rating'] = tempList.copy()

    data = data[data['comment'].apply(lambda xyz: len(xyz) > 0)]
    data = data[data['rating'].apply(lambda rate: float(rate) >= 1)]
    data = data.sample(frac=1).reset_index(drop=True)

    return data

def getTrainData(data, n):
    data = data.sample(frac=1).reset_index(drop=True)
    lenOfData = data.shape[0]

    endIndex = int(lenOfData * n)

    trainSet = data[: endIndex]
    otherSet = data[endIndex:]

    trainSet = trainSet.sample(frac=1).reset_index(drop=True)
    otherSet = otherSet.sample(frac=1).reset_index(drop=True)

    return [trainSet, otherSet]

def getDevAndTestData(data):
    devSet, testSet = getTrainData(data, 0.5).copy()
    
    if devSet.shape[0] >= testSet.shape[0]:
        return [devSet, testSet]
    else:
        return [testSet, devSet]
    
def generateFile(name, data):
    with bz2.BZ2File(name + '.pbz2', 'w') as f:
        cPickle.dump(data, f)
    f.close()
    
def readFile(name):
    with bz2.BZ2File(name, 'rb') as f:
        data = cPickle.load(f)
    f.close()
    return data

def getVectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(data)
    generateFile('vectorizer', vectorizer)
    # vectorizerFile = open('vectorizer', 'ab')
    # pickle.dump(vectorizer, vectorizerFile)
    # vectorizerFile.close()

    return vectorizer

def vectorizeData(data, vectorizer):
    data = vectorizer.transform(data)

    return data

def workOnRatings(ratings):
    tempList = list()
    for r in ratings:
        if r%int(r) <= 0.5:
            tempList.append(int(r))
        else:
            tempList.append(int(r + 1))

    return np.asarray(tempList)


def calAccuracy(actual, pred):
    return accuracy_score(actual, pred)*100
pathToFolder = join(path[0], '/kaggle/input/boardgamegeek-reviews')
#/content/drive/My Drive/Colab Notebooks/boardgamegeek-reviews
pathToFile = join(pathToFolder, 'bgg-13m-reviews.csv')

data = loadAndProcessData(pathToFile)
print(data)

trainDataset, tempDataset = getTrainData(data, 0.95).copy()
devDataset, testDataset = getDevAndTestData(tempDataset)

print('\n\n TRAIN DATASET:::\n')
print(trainDataset)
print('\n\n DEVELOPEMENT DATASET:::\n')
print(devDataset)
print('\n\n TEST DATASET:::')
print(testDataset)


trainComments, trainRatings = [trainDataset['comment'], trainDataset['rating']].copy()
devComments, devRatings = [devDataset['comment'], devDataset['rating']].copy()
testComments, testRatings = [testDataset['comment'], testDataset['rating']].copy()

try:
  vectorizer = readFile('vectorizer.pbz2')
  # vectorizeFile = open('vectorizer', 'rb')
  # vectorizer = pickle.load(vectorizeFile)
  print("Vectorizer read from file")
except (FileNotFoundError):
  vectorizer = getVectorizer(trainComments)
  print("Vectorizer file was not found... calling function")


trainComments, trainRatings = [vectorizeData(trainComments, vectorizer), workOnRatings(trainRatings)]
devComments, devRatings = [vectorizeData(devComments, vectorizer), workOnRatings(devRatings)]
testComments, testRatings = [vectorizeData(testComments, vectorizer), workOnRatings(testRatings)]



nbModel = MultinomialNB(alpha = 1)
nbModel.fit(trainComments, trainRatings)
nbTrainRatingsResults = nbModel.predict(trainComments)
nbDevRatingsResults = nbModel.predict(devComments)
nbTestRatingsResults = nbModel.predict(testComments)
nbTrainAccuracy = calAccuracy(trainRatings, nbTrainRatingsResults)
nbDevAccuracy = calAccuracy(devRatings, nbDevRatingsResults)
nbTestAccuracy = calAccuracy(testRatings, nbTestRatingsResults)
print("\n\nNB accuracy for train dataset: {:0.5f}".format(nbTrainAccuracy))
print("NB accuracy for development dataset: {:0.5f}".format(nbDevAccuracy))
print("NB accuracy for test dataset(FINAL): {:0.5f}".format(nbTestAccuracy))
#hyper....
svmModel1 = LinearSVC(C = 0.1)
svmModel1.fit(trainComments, trainRatings)
svmModel3 = LinearSVC(C = 0.3)
svmModel3.fit(trainComments, trainRatings)
svmModel5 = LinearSVC(C = 0.5)
svmModel5.fit(trainComments, trainRatings)
svmModel7 = LinearSVC(C = 0.7)
svmModel7.fit(trainComments, trainRatings)
svmModel = LinearSVC()
svmModel.fit(trainComments, trainRatings)
svmTrainRatingsResults = svmModel1.predict(trainComments)
svmDevRatingsResults = svmModel1.predict(devComments)
svmTestRatingsResults = svmModel1.predict(testComments)
svmTrainAccuracy = calAccuracy(trainRatings, svmTrainRatingsResults)
svmDevAccuracy = calAccuracy(devRatings, svmDevRatingsResults)
svmTestAccuracy = calAccuracy(testRatings, svmTestRatingsResults)
print("\n\nSVM accuracy for train dataset: {:0.5f}".format(svmTrainAccuracy))
print("SVM accuracy for development dataset: {:0.5f}".format(svmDevAccuracy))
print("SVM accuracy for test dataset(FINAL): {:0.5f}".format(svmTestAccuracy))
svmTrainRatingsResults = svmModel3.predict(trainComments)
svmDevRatingsResults = svmModel3.predict(devComments)
svmTestRatingsResults = svmModel3.predict(testComments)
svmTrainAccuracy = calAccuracy(trainRatings, svmTrainRatingsResults)
svmDevAccuracy = calAccuracy(devRatings, svmDevRatingsResults)
svmTestAccuracy = calAccuracy(testRatings, svmTestRatingsResults)
print("\n\nSVM accuracy for train dataset: {:0.5f}".format(svmTrainAccuracy))
print("SVM accuracy for development dataset: {:0.5f}".format(svmDevAccuracy))
print("SVM accuracy for test dataset(FINAL): {:0.5f}".format(svmTestAccuracy))
svmTrainRatingsResults = svmModel5.predict(trainComments)
svmDevRatingsResults = svmModel5.predict(devComments)
svmTestRatingsResults = svmModel5.predict(testComments)
svmTrainAccuracy = calAccuracy(trainRatings, svmTrainRatingsResults)
svmDevAccuracy = calAccuracy(devRatings, svmDevRatingsResults)
svmTestAccuracy = calAccuracy(testRatings, svmTestRatingsResults)
print("\n\nSVM accuracy for train dataset: {:0.5f}".format(svmTrainAccuracy))
print("SVM accuracy for development dataset: {:0.5f}".format(svmDevAccuracy))
print("SVM accuracy for test dataset(FINAL): {:0.5f}".format(svmTestAccuracy))
svmTrainRatingsResults = svmModel7.predict(trainComments)
svmDevRatingsResults = svmModel7.predict(devComments)
svmTestRatingsResults = svmModel7.predict(testComments)
svmTrainAccuracy = calAccuracy(trainRatings, svmTrainRatingsResults)
svmDevAccuracy = calAccuracy(devRatings, svmDevRatingsResults)
svmTestAccuracy = calAccuracy(testRatings, svmTestRatingsResults)
print("\n\nSVM accuracy for train dataset: {:0.5f}".format(svmTrainAccuracy))
print("SVM accuracy for development dataset: {:0.5f}".format(svmDevAccuracy))
print("SVM accuracy for test dataset(FINAL): {:0.5f}".format(svmTestAccuracy))
svmTrainRatingsResults = svmModel.predict(trainComments)
svmDevRatingsResults = svmModel.predict(devComments)
svmTestRatingsResults = svmModel.predict(testComments)
svmTrainAccuracy = calAccuracy(trainRatings, svmTrainRatingsResults)
svmDevAccuracy = calAccuracy(devRatings, svmDevRatingsResults)
svmTestAccuracy = calAccuracy(testRatings, svmTestRatingsResults)
print("\n\nSVM accuracy for train dataset: {:0.5f}".format(svmTrainAccuracy))
print("SVM accuracy for development dataset: {:0.5f}".format(svmDevAccuracy))
print("SVM accuracy for test dataset(FINAL): {:0.5f}".format(svmTestAccuracy))
try:
  finalSVM = readFile('svmModel.pbz2')
  # vectorizeFile = open('vectorizer', 'rb')
  # vectorizer = pickle.load(vectorizeFile)
  print("Model read from file")
except (FileNotFoundError):
  print("Model file was not found... calling function to create it")
  generateFile('svmModel', svmModel1) 
while True:
  comment = input("Enter comment here...\n")
  if comment == 'q':
    break
  comment = vectorizer.transform([comment])
  rating = svmModel.predict(comment)[0]
  print(rating)