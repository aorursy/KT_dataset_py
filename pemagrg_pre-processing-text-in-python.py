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

def to_lower(text):

    """

    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".

    """

    return ' '.join([w.lower() for w in nltk.word_tokenize(text)])

text = """Harry Potter is the most miserable, lonely boy you can imagine. He's shunned by his relatives, the Dursley's, that have raised him since he was an infant. He's forced to live in the cupboard under the stairs, forced to wear his cousin Dudley's hand-me-down clothes, and forced to go to his neighbour's house when the rest of the family is doing something fun. Yes, he's just about as miserable as you can get."""

print (to_lower(text))
text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

sent_token = nltk.sent_tokenize(text)

print (sent_token)
text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word_tokens = nltk.word_tokenize(text)

print (word_tokens)

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

#is based on The Porter Stemming Algorithm

stopword = stopwords.words('english')

wordnet_lemmatizer = WordNetLemmatizer()

text = "the functions of this fan is great"

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

import re

text = """<head><body>hello world!</body></head>"""

cleaned_text = re.sub('<[^<]+?>','', text)

print (cleaned_text)
text = "There was 200 people standing right next to me at 2pm."

output = ''.join(c for c in text if not c.isdigit())

print(output)
from string import punctuation

def strip_punctuation(s):

    return ''.join(c for c in s if c not in punctuation)

text = "Hello! how are you doing?"

print (strip_punctuation(text))
import nltk

from nltk.corpus import stopwords

stopword = stopwords.words('english')

text = "This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit"

word_tokens = nltk.word_tokenize(text)

removing_stopwords = [word for word in word_tokens if word not in stopword]

print (removing_stopwords)
