# 1. Python open()



import os

with open('../input/poetry/adele.txt','r') as f:

    adele = f.read()
# 2. Web Scraping



import requests

from bs4 import BeautifulSoup
from time import sleep

sleep(5)
url = 'https://www.udacity.com/courses/all'

page = requests.get(url)
# There are two methods of parsing the html page. 1st - Regular Expressions. 2nd - BeautifulSoup

# Regular Expression:



import re
re.findall('[^.,?/''"":;!@#$%&*() ]+', page.text)
pattern = re.compile(r'<,.*?>')

# page_clean = print(pattern.sub(' ', page.text)) # Replaces patterns with space
# Beautiful Soup



soup = BeautifulSoup(page.text, 'html5lib')

print(soup.getText())
soup = BeautifulSoup(page.content, 'html.parser')

for i in soup.select('a'):

    print(i.getText())
#1. Lower Case



adele = adele.lower()



#2. Punctuation Removal

import string

punc = string.punctuation

adele = ''.join([char for char in adele if char not in punc]) #for loops may be inefficient sometimes

adele[:500]
#Removing newlines

#strip function does not remove whitespaces from the middle part of the text. It removes white spaces only from the 

#beginnings and end. Let's use replace to get rid of the newlines.



adele = adele.replace('\n', ' ')

adele[:500]
# using regular expression to get rid of punctuations:

with open('../input/poetry/beatles.txt') as g:

    beatles = g.read()



#Normalization

#Lower case



beatles = beatles.lower()



#Removing Punctuation

import re

beatles = re.sub('[^a-zA-Z0-9\s]','',beatles) #replacing not a-z, A-Z, 0-9 and whitespaces with ''.

beatles = beatles.replace('\n', ' ')
#1. split()



adele_words = adele.split()

adele_words[:5]
#2. nltk



from nltk.tokenize import word_tokenize

adele_nltk_word =word_tokenize(adele)

adele_nltk_word[:2]
from nltk.corpus import stopwords



adele_stopwords_removed = [w for w in adele_nltk_word if w not in stopwords.words('english')]

adele_stopwords_removed[:5]
from nltk import pos_tag

adele_pos_tag = pos_tag(adele_stopwords_removed)

adele_pos_tag[:5]
from nltk import ne_chunk

adele_named_entity = ne_chunk(adele_pos_tag)
print(ne_chunk(pos_tag(word_tokenize("Don't Bullshit me"))))
print(ne_chunk(pos_tag(word_tokenize("This is bullshit"))))
from nltk.stem import LancasterStemmer

from nltk.stem import PorterStemmer



ps = PorterStemmer()

list(map(ps.stem,adele_stopwords_removed))