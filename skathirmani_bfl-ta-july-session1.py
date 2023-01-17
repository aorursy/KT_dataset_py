# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import nltk

amazon = pd.read_csv('/kaggle/input/trainings/amazon_reviews_big.csv')
amazon.head()
## Numerical: mean, median or distribution analysis
## Categorical: mode or frequency analysis
## Text column: frequency analysis (bag of word analysis)
# bag of word analysis: Frequency of each term and idenitify frequently appearing term
# bar chart, word cloud
from wordcloud import WordCloud
%matplotlib inline
import matplotlib.pyplot as plt
docs = amazon['reviewText']
sample = 'i like india i love coffee and the'
wc = WordCloud(background_color='white').generate(sample)
plt.imshow(wc)
docs = amazon['reviewText'].fillna('Not available')
wc = WordCloud(background_color='white').generate(' '.join(docs))
plt.imshow(wc)
### Why "i" is coming
# Lowercase conversation
docs = amazon['reviewText'].fillna('Not available').str.lower()
docs.head()
## Tokenization: Identifying token/terms
doc = docs[0]
doc
terms = doc.split(' ')
len(terms)
from nltk.tokenize import word_tokenize

terms = word_tokenize(doc)
from nltk.tokenize import RegexpTokenizer
pattern = '\w+'
regex_tokenizer = RegexpTokenizer(pattern)
tokens = regex_tokenizer.tokenize(doc)
from nltk.tokenize import TweetTokenizer
doc = '@sam i like this product very much 😍😍😍😍 #iphone'
#print(nltk.word_tokenize(doc))
tweet_tokenizer = TweetTokenizer()
tokens = tweet_tokenizer.tokenize(doc)
print(tokens)
docs_cleaned = []

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

stopwords = nltk.corpus.stopwords.words('english')
custom_stopwords = ['work', 'well']
stopwords.extend(custom_stopwords)
for doc in docs.str.lower().head(10000):
    tokens = regex_tokenizer.tokenize(doc)
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
    doc_cleaned = ' '.join(tokens)
    docs_cleaned.append(doc_cleaned)
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('visited', pos='v')
nltk.pos_tag(['visited'])
import spacy
nlp = spacy.load("en_core_web_sm")
doc_nlp = nlp('i visited this place yesterday')
for token in doc_nlp:
    print(token, token.pos_, token.lemma_)
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmer.stem('organization')
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern='[a-z]{,10}', min_df=5, stop_words='english').fit(docs_cleaned)
vocab = vectorizer.get_feature_names()
len(vocab)
dtm = vectorizer.transform(docs_cleaned)
dtm
df_dtm = pd.DataFrame(dtm.toarray(), columns=vocab)
df_dtm.shape
zeros = (df_dtm == 0).sum().sum()
values = df_dtm.shape[0] * df_dtm.shape[1]
zeros/values*100
values - zeros
# Column sum: term wise frequency
df_dtm.sum().sort_values(ascending=False).drop(['']).head(20)[::-1].plot.barh(figsize=(5,10))
# Row sum
df_dtm.sum(axis=1).sort_values(ascending=False).head(5)
#amazon['reviewText'].iloc[6364]
term = 'camera'
df_dtm['camera'].plot.box()
temp = df_dtm[['camera', 'price']]
temp[(temp['camera']!=0) & (temp['price']!=0)].head()
## Bigrams & Trigrams
vectorizer = CountVectorizer(min_df=5,
                             stop_words='english',
                             ngram_range=(2,2)).fit(docs_cleaned)
vocab = vectorizer.get_feature_names()
dtm = vectorizer.transform(docs_cleaned)
df_dtm = pd.DataFrame(dtm.toarray(), columns=vocab)
df_dtm.sum().sort_values(ascending=False).head(10)[::-1].plot.barh(figsize=(5,10))
