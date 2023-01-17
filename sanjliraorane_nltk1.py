# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import nltk

#natural language toolkit
from nltk.book import*

#9 types of text available in NLTK book you may choose any text as per your interest
# display all occurences of word in piece of text along with context

text1.concordance("country")
text2.concordance("small")
# returns a list of words that appaer in similar context usually synonyms

text1.similar("country")
text2.similar("small")
# returns contexts shared by 2 words

text2.common_contexts(['country','small'])
# print plot of all occurences of the word relative to begining of the text

text1.dispersion_plot(['country','small'])
from nltk.tokenize import word_tokenize, sent_tokenize

text="Gud morning, have a nice day. "

sents=sent_tokenize(text)

print(sents)
l= nltk.word_tokenize(text)

print(l)
# Let's filter out stopwords (words that are very common like 'was', 'a', 'as etc)

from nltk.corpus import stopwords 

from string import punctuation

customStopWords=set(stopwords.words('english')+list(punctuation))

#Notice how we made the stopwords a set



wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]

print(wordsWOStopwords)
text2="BVCOEW is a good college."

# 'close' appears in different morphological forms here, stemming will reduce all forms of the word 'close' to its root

# NLTK has multiple stemmers based on different rules/algorithms. Stemming is also known as lemmatization. 

from nltk.stem.lancaster import LancasterStemmer

st=LancasterStemmer()

stemmedWords=[st.stem(word) for word in word_tokenize(text2)]

print(stemmedWords)
#NLTK has functionality to automatically tag words as nouns, verbs, conjunctions etc

nltk.pos_tag(word_tokenize(text2))