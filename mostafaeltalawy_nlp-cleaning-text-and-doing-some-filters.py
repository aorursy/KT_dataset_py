#importing lib
import gc
import re
import string
import operator
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import STOPWORDS

#text loading 
text=" #ourhashtag , this is our new data , and i will start training on it ! training data is must , As we need to work on it tomorrow . are you ready ? @mostafa "
#counting words in text and doing some other  calculations 
word_splitting=str(text).split() #split()  it will split every word after each space...if we make it like split(".").. it will split words after each dot
print(word_splitting)
len(word_splitting)
#counting unique words 
unique_words=set(word_splitting)
print(unique_words)
len(unique_words)
#counting stop words
stop_words=[]
for w in word_splitting:
    if w in STOPWORDS:
        stop_words.append(w)
print(stop_words) 
len(stop_words)
#deleting stop words from our words
filtered_words=[]
for w in word_splitting:
    if w not in STOPWORDS:
        filtered_words.append(w)
print(filtered_words)
len(filtered_words)
print('punctuations are  ',string.punctuation)
# punctuation_count

punctuations=[]
for w in filtered_words: #if we searched in all text before splitting it will give us also # and @
    if w in string.punctuation:
        punctuations.append(w)
print(punctuations)        
print( 'no of punctuations is = ',len(punctuations))
print("==========")

punctuations1=[]
for w in text:
    if w in string.punctuation:
        punctuations1.append(w)
print(punctuations1)        
print( 'no of punctuations is = ',len(punctuations1))
print("==========")

#removing punctuations from splitting test
text_words=[]
for w in filtered_words:
    if w not in string.punctuation:
        text_words.append(w)
print('filltered words are = ',text_words)
print("------------------------")
print('no. of filtered words are = ',len(text_words) )
#getting unique words only

unique_words=set(text_words)
print(unique_words)
print("no. of unique_words are = ",len(unique_words))
#counting and extracting hashtage and mentioned persons

hash_tag=[]
for char in str(text):
    if char =='#':
        hash_tag.append(char)
print( 'no of hashtag is = ',len(hash_tag))
print("==========")

mentioned=[]
for char in str(text):
    if char == '@':
        mentioned.append(char)
print( 'no of mentioned persons is = ',len(mentioned))
#if you want to find words starts with any characters 
wordstartwith = re.findall(r'\ba\w+', text) 
print("words start with a are  = ",wordstartwith)
#  if we want to search for words start with a and A 
wordstartwith1 = re.findall(r'\b[Aa]\w+', text) 
print("words start with a and A are  =", wordstartwith1)
#finding hashtages and mentioned in text
hashtag =list(filter(lambda word: word[0]=='#', text.split()))
print("hashtags are : ",hashtag)
mentioned =list(filter(lambda word: word[0]=='@', text.split()))
print("mentioned persons are : ",mentioned)

