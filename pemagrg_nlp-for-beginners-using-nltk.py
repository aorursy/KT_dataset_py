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

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

lower_text = text.lower()

print (lower_text)

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word_tokens = nltk.word_tokenize(text)

print (word_tokens)

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

sent_token = nltk.sent_tokenize(text)

print (sent_token)
import nltk

from nltk.corpus import stopwords

stopword = stopwords.words('english')

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word_tokens = nltk.word_tokenize(text)

removing_stopwords = [word for word in word_tokens if word not in stopword]

print (removing_stopwords)
import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

#is based on The Porter Stemming Algorithm

stopword = stopwords.words('english')

wordnet_lemmatizer = WordNetLemmatizer()



text = "the dogs are barking outside. Are the cats in the garden?"

word_tokens = nltk.word_tokenize(text)

lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]

print (lemmatized_word)
import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

#is based on The Porter Stemming Algorithm

stopword = stopwords.words('english')

snowball_stemmer = SnowballStemmer('english')

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word_tokens = nltk.word_tokenize(text)

stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]

print (stemmed_word)
import nltk

from nltk import FreqDist

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word = nltk.word_tokenize(text.lower())

freq = FreqDist(word)

print (freq.most_common(5))
import nltk

text = "the dogs are barking outside."

word = nltk.word_tokenize(text)

pos_tag = nltk.pos_tag(word)

print (pos_tag)

import nltk

text = "who is Barrack Obama"

word = nltk.word_tokenize(text)

pos_tag = nltk.pos_tag(word)

chunk = nltk.ne_chunk(pos_tag)

NE = [ " ".join(w for w, t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]

print (NE)