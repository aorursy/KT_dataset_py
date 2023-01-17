# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from nltk.tokenize import word_tokenize

from nltk.text import Text
my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees.See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."

tokens = word_tokenize(my_string)



tokens = [word.lower() for word in tokens]

tokens[:10]
t = Text(tokens)

t
t.concordance('uckers') # concordance() is a method of the Text class of NLTK. It finds words and displays a context window. Word matching is not case-sensitive.

# concordance() is defined as follows: concordance(self, word, width=79, lines=25). Note default values for optional params.
from nltk.tokenize import sent_tokenize

text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.

The sky is pinkish-blue. You shouldn't eat cardboard"""

tokenized_text=sent_tokenize(text)

print(tokenized_text)
from nltk.tokenize import word_tokenize

tokenized_word=word_tokenize(text)

print(tokenized_word)
from nltk.probability import FreqDist

fdist = FreqDist(tokenized_word)

print(fdist)
fdist.most_common(2)
# Frequency Distribution Plot

import matplotlib.pyplot as plt

fdist.plot(30,cumulative=False)

plt.show()
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))

print(stop_words)
# Stemming

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize



ps = PorterStemmer()



stemmed_words=[]

for w in filtered_sent:

    stemmed_words.append(ps.stem(w))



print("Filtered Sentence:",filtered_sent)

print("Stemmed Sentence:",stemmed_words)
#Lexicon Normalization

#performing stemming and Lemmatization



from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()



from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()



word = "flying"

print("Lemmatized Word:",lem.lemmatize(word,"v"))

print("Stemmed Word:",stem.stem(word))
sent = "Albert Einstein was born in Ulm, Germany in 1879."