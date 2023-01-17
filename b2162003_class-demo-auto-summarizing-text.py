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
articleURL="https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/"
response = requests.get(articleURL)
response.encoding = 'utf-8'
data = response.text
soup = BeautifulSoup(data)
print(soup)


soup.find('article').text
text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
text
text.encode('ascii', 'ignore')
print(text)
                                               
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
sents = sent_tokenize(text)
sents
word_sent = word_tokenize(text.lower())
word_sent
_stopwords = set(stopwords.words('english') + list(punctuation))
_stopwords
# Filter out stopword
word_sent=[word for word in word_sent if word not in _stopwords]
word_sent
from nltk.probability import FreqDist
freq = FreqDist(word_sent)
freq

from heapq import nlargest
nlargest(10, freq, key=freq.get)
# We want to create a signifcant score ordered by highest frequency
from collections import defaultdict
ranking = defaultdict(int)
for i,sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
ranking
# Top 4 Sentences
sents_idx = nlargest(4, ranking, key=ranking.get)
sents_idx
[sents[j] for j in sorted(sents_idx)]