# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import requests
from bs4 import BeautifulSoup

from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation


# Any results you write to the current directory are saved as output.
#Interchange this url with any other one of choice, as part of number 1
#articleURL="https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/"
articleURL = input("Enter your URL: ")
#This block parses the html of the page to separate it from the text
response = requests.get(articleURL)
response.encoding = 'utf-8'
data = response.text
soup = BeautifulSoup(data, features ="html.parser")
print(soup)

#This block checks to verify the input url contains an article tag
#It then prints to let the user know if an article tag was or was not found

for tag in soup.find_all('article'):
    if(tag.name == 'article'):
        print("Article tag found in your html page")
    else:
        print("No article tag found")
#soup.find('article').text
#returns the same thing as soup.find('article').text
text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
print(text)

#this block cleans up the text from the above blcok called sents
#it removes all new line tags and 's so word_sent is cleaner
text = text.replace("\n", " ")
text = text.replace("'s", "")
text = text.replace(".",". ")

print(text)
#this just prints text again in ascii
text.encode('ascii', 'ignore')
print(text)                                               
#this block breaks the article into sentences
sents = sent_tokenize(text)
sents
word_sent = word_tokenize(text.lower())
word_sent
_stopwords = set(stopwords.words('english') + list(punctuation))
_stopwords
# Filter out stopword
word_sent=[word for word in word_sent if word not in _stopwords]
word_sent
freq = FreqDist(word_sent)
freq

#this block gets the ten most frequently used words
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
# Top 3 Sentences
sents_idx = nlargest(3, ranking, key=ranking.get)
sents_idx
#this block prints the summarized strings
formattedOutput = [sents[j] for j in sorted(sents_idx)]
print("Summary Below: ")
print(formattedOutput[0])
print(formattedOutput[1])
print(formattedOutput[2])