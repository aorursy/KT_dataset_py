

#Importing the essential libraries

#Beautiful Soup is a Python library for pulling data out of HTML and XML files

#The Natural Language Toolkit



import requests

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer 

from bs4 import BeautifulSoup

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import random

from wordcloud import WordCloud

from html.parser import HTMLParser



import bs4 as bs

import urllib.request

import re
#we are using request package to make a GET request for the website, which means we're getting data from it.

#It is a "Times Of India" article on the economic impact of COVID-19



r=requests.get('https://timesofindia.indiatimes.com/blogs/transforming-india/episode-11-economic-impact-of-coronavirus/')


#Setting the correct text encoding of the HTML page

r.encoding = 'utf-8'
#Extracting the HTML from the request object

html = r.text
# Printing the first 500 characters in html

print(html[:500])


# Creating a BeautifulSoup object from the HTML

soup = BeautifulSoup(html)



# Getting the text out of the soup

text = soup.get_text()
#total length

len(text)

#We are taking this range, as this contains the actual news article.

text=text[48700:70000]
text[10000:12000]
#The text of the novel contains a lot of unwanten stuff, we need to remove them

#We will start by tokenizing the text, that is, remove everything that isn't a word (whitespace, punctuation, etc.) 

#Then then split the text into a list of words
#CLEANING THE PUNCTUATION
len(text)
import string

string.punctuation
text_nopunct=''



text_nopunct= "".join([char for char in text if char not in string.punctuation])
len(text_nopunct)
#We see a large part has been cleaned
#Creating the tokenizer

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
#Tokenizing the text

tokens = tokenizer.tokenize(text_nopunct)
len(tokens)
print(tokens[0:20])
#now we shall make everything lowercase for uniformity

#to hold the new lower case words



words = []



# Looping through the tokens and make them lower case

for word in tokens:

    words.append(word.lower())
print(words[0:500])
#Stop words are generally the most common words in a language.

#English stop words from nltk.



stopwords = nltk.corpus.stopwords.words('english')
words_new = []



#Now we need to remove the stop words from the words variable

#Appending to words_new all words that are in words but not in sw



for word in words:

    if word not in stopwords:

        words_new.append(word)
len(words_new)
from nltk.stem import WordNetLemmatizer 

  

wn = WordNetLemmatizer() 
lem_words=[]



for word in words_new:

    word=wn.lemmatize(word)

    lem_words.append(word)

    

len(lem_words)
len(words_new)
same=0

diff=0



for i in range(0,1832):

    if(lem_words[i]==words_new[i]):

        same=same+1

    elif(lem_words[i]!=words_new[i]):

        diff=diff+1
print('Number of words Lemmatized=', diff)

print('Number of words not Lemmatized=', same)
#This shows that lemmatization helps to process the data efficiently.

#It is used in- 

# 1. Comprehensive retrieval systems like search engines.

# 2. Compact indexing
#The frequency distribution of the words

freq_dist = nltk.FreqDist(lem_words)
#Frequency Distribution Plot

plt.subplots(figsize=(20,12))

freq_dist.plot(50)
#converting into string



res=' '.join([i for i in lem_words if not i.isdigit()])
plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          background_color='black',

                          max_words=100,

                          width=1400,

                          height=1200

                         ).generate(res)





plt.imshow(wordcloud)

plt.title('Economic News (100 words)')

plt.axis('off')

plt.show()
plt.subplots(figsize=(16,10))

wordcloud = WordCloud(

                          background_color='black',

                          max_words=200,

                          width=1400,

                          height=1200

                         ).generate(res)





plt.imshow(wordcloud)

plt.title('Economic News (200 words)')

plt.axis('off')

plt.show()
# Removing Square Brackets and Extra Spaces

clean_text = re.sub(r'\[[0-9]*\]', ' ', text)

clean_text = re.sub(r'\s+', ' ', clean_text)
clean_text[0:500]
#We need to tokenize the article into sentences

#Sentence tokenization



sentence_list = nltk.sent_tokenize(clean_text)
sentence_list
#Weighted Frequency of Occurrence



stopwords = nltk.corpus.stopwords.words('english')



word_frequencies = {}

for word in nltk.word_tokenize(clean_text):

    if word not in stopwords:

        if word not in word_frequencies.keys():

            word_frequencies[word] = 1

        else:

            word_frequencies[word] += 1
type(word_frequencies)
maximum_frequncy = max(word_frequencies.values())



for word in word_frequencies.keys():

    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
sentence_scores = {}

for sent in sentence_list:

    for word in nltk.word_tokenize(sent.lower()):

        if word in word_frequencies.keys():

            if len(sent.split(' ')) < 30:

                if sent not in sentence_scores.keys():

                    sentence_scores[sent] = word_frequencies[word]

                else:

                    sentence_scores[sent] += word_frequencies[word]
sentence_scores
import heapq

summary_sentences = heapq.nlargest(20, sentence_scores, key=sentence_scores.get)



summary = ' '.join(summary_sentences)

print(summary)
#Thank You