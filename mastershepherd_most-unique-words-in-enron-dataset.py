%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from math import *



import nltk

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk.corpus import brown

from nltk.corpus import stopwords

#This needs to be changed to whatever your directory i

masterDF = pd.read_csv("../input/emails.csv")
messageList = masterDF['message'].tolist()

#print messageList[11]



bodyList = []



#Janky first attempt at a split!

for message in messageList:

    #Split at the filename

    firstSplit = message.split("X-FileName: ", 1)[1]

    #Get everything after the file extension

    secondSplit = firstSplit.split(".")

    #Some error checking if the file type isn't included

    if len(secondSplit) > 1:

        secondSplit = secondSplit[1]

    body =  ''.join(secondSplit)[4:]

    bodyList.append(body)
#Join all of this text together

textBlob = ''.join(bodyList)



textTokenized = word_tokenize(textBlob)

textFreqDist = FreqDist(textTokenized)
#Get a frequency distribution from Brown

brownFreqDist = FreqDist(i.lower() for i in brown.words())
uniquenessList = []



#Compare the occurance of a word to its occurance in the Brown Dataset

for word in textFreqDist:

    brownOccurances = brownFreqDist[word]

    textOccurances = textFreqDist[word]

    if brownOccurances > 5 and textOccurances > 5 and word.isalpha():

        uniquenessList.append((word, log10(float(textOccurances) / float(brownOccurances))))
uniquenessList.sort(key=lambda tup: -tup[1])



for i in range(10):

    print("(%s, %f)" % (uniquenessList[i][0], uniquenessList[i][1]))