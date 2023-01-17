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
import nltk
#Bigrams represent a set of two consecutive words appearing in a text.

#bigrams function is called on tokenized words, as shown in the following example, to obtain bigrams.

s = 'Python is an awesome language.'

tokens = nltk.word_tokenize(s)
list(nltk.bigrams(tokens))
#Now let's find out three frequently occurring bigrams, present in english-kjv collection of genesis corpus.

#Let's consider only those bigrams, whose words are having a length greater than 5

#Working with Genesis Corpus

from nltk.corpus import genesis

eng_tokens = genesis.words('english-kjv.txt')
eng_bigrams = nltk.bigrams(eng_tokens)

#bigrams with length greater than or equal to 5

filtered_bigrams = [ (w1, w2) for w1, w2 in eng_bigrams if len(w1) >=5 and len(w2) >= 5 ]
#After computing bi-grams, the following code computes frequency distribution and displays three most frequent bigrams.

eng_bifreq = nltk.FreqDist(filtered_bigrams)

#Most Common top 3

eng_bifreq.most_common(3)
eng_tokens = genesis.words('english-kjv.txt')

eng_bigrams = nltk.bigrams(eng_tokens)

#Now let's see an example which determines the two most frequent words occurring after 'living' are determined.

eng_cfd = nltk.ConditionalFreqDist(eng_bigrams)

eng_cfd['living'].most_common(2)
#Now let's define a function named generate, which returns words occurring frequently after a given word.

def generate(cfd, word, n=5):

    n_words = []

    for i in range(n):

         n_words.append(word)

         word = cfd[word].max()

    return n_words    
#Generating Most Frequent Next Word

generate(eng_cfd, 'living')
#Working with Brown Corpus
#Import text corpus brown

#Extract the list of words associated with text collections belonging to news genre. Store the result in variable news_words.

#Convert each word of list news_words into lower case and store the result in lc_news_words.

#Compute length of each word present in list lc_news_words and store the result in list len_news_words

#Compute bigrams of list len_news_words. Store the result in variable news_len_bigrams.

#Compute the conditional frequency of news_len_bigrams, where condition and event refers to length of a words. Store the result in cfd_news

#Determine the frequency of 6-letter words appearing next to a 4-letter word.
from nltk.corpus import brown

list((genre, word) for genre in brown.categories() for word in brown.words(categories=genre))
from nltk.corpus import brown

news_words = brown.words(categories='news')

lc_news_words = map(lambda x:x.lower(),news_words)

len_news_words = map(lambda x:len(x),news_words)

news_len_bigrams = list(nltk.bigrams(len_news_words))

cfd_news = nltk.ConditionalFreqDist(news_len_bigrams)

print([(i,j) for i,j in cfd_news[4].most_common(25) if i == 6][0][1])
cfd_news[4].most_common(25)
#Compute bigrams of list lc_news_words and store in variable lc_news_bigrams.

#From lc_news_bigrams filter those bigrams, whose both words contain only alphabet characters. Store the result in lc_news_alpha_bigrams.

#Extract the list of words associated with corpus stopwords. Store the result in stop_words.

#Convert each word of list stop_words into lower case and store the result in lc_stop_words.

#Filter only those bigrams from lc_news_alpha_bigrams, whose words are not part of lc_stop_words. Store the result in lc_news_alpha_nonstop_bigrams.

#Print the total number of filtered bigrams.
#Promblem 1

import nltk

from nltk.corpus import brown

news_words = brown.words(categories='news')

lc_news_words = map(lambda x:x.lower(),news_words)

len_news_words = map(lambda x:len(x),news_words)

news_len_bigrams = list(nltk.bigrams(len_news_words))

cfd_news = nltk.ConditionalFreqDist(news_len_bigrams)

print([(i,j) for i,j in cfd_news[4].most_common(25) if i == 6][0][1])

lc_news_words = map(lambda x:x.lower(),news_words)

#Problem 2

#Bigrams of News Words

lc_news_bigrams = list(nltk.bigrams(lc_news_words))

#Filtered BiGrams

lc_news_alpha_bigrams = [ (w1, w2) for w1, w2 in lc_news_bigrams if w1.isalpha()==True and w2.isalpha()==True]

from nltk.corpus import stopwords

#Stopwords

stop_words = stopwords.words()

lc_stop_words = [w.lower() for w in stop_words]

#List of Bigrams not part of Stop Words

lc_news_alpha_nonstop_bigrams = [(w1, w2) for w1, w2 in lc_news_alpha_bigrams if not w1 in lc_stop_words and not w2 in lc_stop_words]

#Total Number of Bigrams

print(len(lc_news_alpha_bigrams) + len(lc_news_alpha_nonstop_bigrams))
s = 'Python is cool!!!'

tokens = nltk.word_tokenize(s)

list(nltk.trigrams(tokens))
#The following example displays a list of four consecutive words appearing in the text s.

s = 'Python is an awesome language.'

tokens = nltk.word_tokenize(s)

list(nltk.ngrams(tokens, 4))
#A collocation is a pair of words that occur together, very often.

#For example, red wine is a collocation.

#One characteristic of a collocation is that the words in it cannot be substituted with words having similar senses.

#For example, the combination maroon wine sounds odd.

#Generating Collocations:

from nltk.corpus import genesis

tokens = genesis.words('english-kjv.txt')

gen_text = nltk.Text(tokens)

gen_text.collocations()
from nltk.book import text6
eng_bigrams = nltk.bigrams(text6.tokens)

eng_bifreq = nltk.FreqDist(eng_bigrams)
#Frequency of Bigram ('BLACK', 'KNIGHT')

eng_bifreq[('BLACK', 'KNIGHT')]
#Frequency of Bigram ('HEAD', 'KNIGHT')

eng_bifreq[('HEAD', 'KNIGHT')]
#Frequency of Bigram ('clop', 'clop')

eng_bifreq[('clop', 'clop')]
#most frequent words occurring after 'Holy'

eng_tokens = text6.tokens

eng_bigrams = nltk.bigrams(eng_tokens)

#Now let's see an example which determines the two most frequent words occurring after 'living' are determined.

eng_cfd = nltk.ConditionalFreqDist(eng_bigrams)

eng_cfd['Holy'].most_common(2)