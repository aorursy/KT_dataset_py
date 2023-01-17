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
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
articleURL="https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/"
def articleCheck(articleURL): 
    response = requests.get(articleURL)
    response.encoding = 'utf-8'
    data = response.text
    soup = BeautifulSoup(data)
    if soup.find('article') is None:
        print ("No valid article tag!")
    else :
        return soup.find('article')
validArticle = articleCheck(articleURL)
def summarizeArticle(validArticle):
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    text.encode('ascii', 'ignore')
    text = text.replace('\n', ' ').replace('\r', '').replace('  ','').replace('\' s','\'s')
    sents = sent_tokenize(text)
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))
    word_sent=[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    nlargest(10, freq, key=freq.get)
    ranking = defaultdict(int)
    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
    sents_idx = nlargest(4, ranking, key=ranking.get)
    print([sents[j] for j in sorted(sents_idx)])
summarizeArticle(validArticle)
#Regardless of what I do, the library is extracting "'s" as a word. I tried removing preceding spaces to
#connect it to whatever word it belongs with but it will not stop counting it as a word. As far as the 
#metrics go, there is no way to compare it to anything because the scikit metrics library requires
#an "answer"


            
    
    




    


                                               












