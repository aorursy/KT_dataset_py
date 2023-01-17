##Loading All the libraries

import os

import spacy

import numpy as np

import requests

from nltk import pos_tag

# coding: utf-8

from spacy import displacy

import re

import string

import nltk

from requests import get



from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from bs4 import BeautifulSoup

import en_core_web_sm

nlp = en_core_web_sm.load()

import pandas as pd

from nltk.corpus import stopwords

import spacy
#loading Urls

url1 = 'https://csee.essex.ac.uk/staff/udo/index.html'

url2= 'https://www.essex.ac.uk/departments/computer-science-and-electronic-engineering'

url =[url1 , url2]

docid = 0

document = []
#Function for Formating html parsing and returning text

def parsing(url):

    res = requests.get(url)

    html = res.text

    soup = BeautifulSoup(html, 'html5lib')

    for script in soup(["script", "style", 'aside']):

        script.extract()

    text = " ".join(re.split(r'[\n\t]+', soup.get_text()))

    return text
#html parsing

text =[]

for i in url:

    text.append(parsing(i))
#Saving output to a file

with open('Htmlprasing1.txt', 'w') as f:

    print(text[0], file=f)

with open('Htmlprasing2.txt', 'w') as f:

    print(text[1], file=f)
#Function for Pre-processing-Removing stop words and lemmatization

def text_processing(text):

    text = text.lower()

    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)

    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    filtered_words = ' '.join(map(str, filtered_words))

    return filtered_words
#Text pre-processing

preprocessed_data1 = text_processing(text[0])

preprocessed_data2 = text_processing(text[1])
#Saving output to a file

with open('preprocessed_data1.txt', 'w') as f:

    print(preprocessed_data1, file=f)

with open('preprocessed_data2.txt', 'w') as f:

    print(preprocessed_data2, file=f)
#Function for word tokenization 

def tokenize(text):

    words = word_tokenize(text)

    return words
#word tokenization

wordtokens =[]

for i in text:

    wordtokens.append(tokenize(i))
#Saving output to a file

with open('word_tokens1.txt', 'w') as f:

    print(wordtokens[0], file=f)

with open('word_tokens2.txt', 'w') as f:

    print(wordtokens[1], file=f)
#Function for Sentence tokenizing

def senttokenize(text):

    words = sent_tokenize(text)

    return words
#Sentence tokenization

sentencetokens =[]

for i in text:

    sentencetokens.append(senttokenize(i))
#Saving output to a file

with open('sent_tokenize1.txt', 'w') as f:

    print(sentencetokens[0], file=f)

with open('sent_tokenize2.txt', 'w') as f:

    print(sentencetokens[1], file=f)
#Function for Part Of Speech tagging

def pos_tagging(text):

    words = word_tokenize(text)

    return pos_tag(words)
#Part Of Speech tagging

postags =[]

for i in text:

    postags.append(pos_tagging(i))
#Saving output to a file

with open('POS_tagging1.txt', 'w') as f:

    print(postags[0], file=f)

with open('POS_tagging2.txt', 'w') as f:

    print(postags[1], file=f)
#Function for Pre-processing-Removing stop words and lemmatization

def text_processing(text):

    text = text.lower()

    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)

    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    filtered_words = ' '.join(map(str, filtered_words))

    return filtered_words
#Text pre-processing

preprocessed_data1 = text_processing(text[0])

preprocessed_data2 = text_processing(text[1])
#Saving output to a file

with open('preprocessed_data1.txt', 'w') as f:

    print(preprocessed_data1, file=f)

with open('preprocessed_data2.txt', 'w') as f:

    print(preprocessed_data2, file=f)
#Creating corpus

corpus=[preprocessed_data1,preprocessed_data2]
#Saving output to a file

with open('corpus.txt', 'w') as f:

    print(corpus, file=f)
#Indexing

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)
#Top 50 words

n = 50

for i in range(0,2):

    feature_array = np.array(vectorizer.get_feature_names())

    tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:n]

    print("Top words in doc {}".format(i+1))

    print(top_n)

    print("\n")
#Saving output to a file

with open('top_50words.txt', 'w') as f:

    print(top_n, file=f)
#Calculating feature scores

Feature_scores=dict(zip(vectorizer.get_feature_names(),X.data))
#Saving output to a file

with open('Feature_scores.txt', 'w') as f:

    print(Feature_scores, file=f)
#Function for NER


nlp = spacy.load('en_core_web_sm')
document1 = nlp(text[0])

document2 = nlp(text[1])
entity_listdoc1 = []



for ent in document1.ents:

    entity_listdoc1.append([ent, ent.label_])
entity_listdoc2 = []



for ent in document2.ents:

    entity_listdoc2.append([ent, ent.label_])
#Saving output to a file

with open('NER1.txt', 'w') as f:

    print(entity_listdoc1, file=f)
#Saving output to a file

with open('NER2.txt', 'w') as f:

    print(entity_listdoc2, file=f)