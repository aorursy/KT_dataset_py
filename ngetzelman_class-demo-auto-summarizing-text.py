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
import requests
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
#instantiate all global variables
soup = ""
text = ""
sents = ""
word_sent = ""
_stopwords = ""
word_sent = ""
freq = ""
ranking = ""
sents_idx = ""
#

def inputURL():
    
    global soup
    articleURL="https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/"
    response = requests.get(articleURL)
    response.encoding = 'utf-8'
    data = response.text
    soup = BeautifulSoup(data)
    
#scrub text

def cleanText():
    global text
    
    #initiallizes the text variable
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    
    text = text.replace("\n", "")  #This takes out all \n in the text
    text = text.replace(".", ". ")
    text = text.replace("\'s", "") #this takes out the 's that has been showing up in the highest frequency table

    text.encode('ascii', 'ignore')
#create values for learner

def learnerVars():
    global sents
    global word_sent
    global _stopwords
    global word_sent
    global freq
    
    sents = sent_tokenize(text)
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))
    word_sent=[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
# function for word rankings

def findRankings():
    global ranking
    
    ranking = defaultdict(int)
    
    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]

#find top three sentences 

def topThree():
    global sents_idx
    
    sents_idx = nlargest(3, ranking, key=ranking.get)
    newOutput = [sents[j] for j in sorted(sents_idx)]
    
    print("\nYour Summary:\n")
    
    #print final summary
    for j in newOutput:
        print(j)
#Final run through of all functions in one block

inputURL()
cleanText()
learnerVars()
findRankings()
topThree()