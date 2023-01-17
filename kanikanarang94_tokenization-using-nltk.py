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
import pandas as pd
data = pd.read_csv("../input/GrammarandProductReviews.csv")
data.head()
data["reviews.text"][0]
import nltk
nltk.word_tokenize(data["reviews.text"][0])
data["reviews.text"][0]
# Tokenize using the white spaces
nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])
# Tokenize using Punctuations
nltk.tokenize.WordPunctTokenizer().tokenize(data["reviews.text"][0])
#Tokenization using grammer rules
nltk.tokenize.TreebankWordTokenizer().tokenize(data["reviews.text"][0])
#Original Sentence
data["reviews.text"][0]
#STEMMING
words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#porter's stemmer
porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
#SnowBall stemmer
snowballStemmedWords = [nltk.stem.SnowballStemmer("english").stem(word) for word in words]
df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)
df
#LEMMATIZATION
words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["reviews.text"][0])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
#WordNet Lemmatization
wordNetLemmatizedWords = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)
df