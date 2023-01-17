# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import nltk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from nltk.book import *

text1.concordance("monstrous")

text2.concordance("monstrous")

text2.similar("monstrous")

text2.common_contexts(["monstrous","very"])
text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])

text2.dispersion_plot(["happy","sad"])

text1.dispersion_plot(["happy","sad"])
from nltk.tokenize import word_tokenize, sent_tokenize
text="Mary had a little lamb. Her fleece was white as snow"
sents=sent_tokenize(text)
print(sents)
words=[word_tokenize(sent) for sent in sents]

print(words)
from nltk.corpus import stopwords 
from string import punctuation
customStopWords=set(stopwords.words('english')+list(punctuation))
wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWOStopwords)

text2="Mary closed on closing night when she was in the mood to close."
from nltk.stem.lancaster import LancasterStemmer
st=LancasterStemmer()
stemmedWords=[st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)
nltk.pos_tag(word_tokenize(text2))