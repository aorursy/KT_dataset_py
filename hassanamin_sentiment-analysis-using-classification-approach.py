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
import nltk
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

print("2 most common :- ",fdist.most_common(2))
# Frequency Distribution Plot

import matplotlib.pyplot as plt

fdist.plot(30,cumulative=False)

plt.show()
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))

print(stop_words)
from nltk.tokenize import word_tokenize

text="""Hello Mr. Smith, how are you doing today?"""

tokenized_sent=word_tokenize(text)

################3

filtered_sent=[]

for w in tokenized_sent:

    if w not in stop_words:

        filtered_sent.append(w)

print("Tokenized Sentence:",tokenized_sent)

print("Filterd Sentence:",filtered_sent)
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
# Sentence Tokenization

sent = "Albert Einstein was born in Ulm, Germany in 1879."

tokens=nltk.word_tokenize(sent)

print(tokens)

# POS Tagging

nltk.pos_tag(tokens)
# Import pandas

import pandas as pd
data=pd.read_csv('../input/train.tsv', sep='\t')

print(data.head())
print(data.info())
print(data.Sentiment.value_counts())
import matplotlib.pyplot as plt

Sentiment_count=data.groupby('Sentiment').count()

plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])

plt.xlabel('Review Sentiments')

plt.ylabel('Number of Review')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

#tokenizer to remove unwanted elements from out data like symbols and numbers

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(data['Phrase'])

print(type(text_counts))

print(text_counts.shape)

#print(text_counts.toarray())

print(text_counts.toarray())

#print(cv.vocabulary_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.3, random_state=1)
from sklearn.naive_bayes import MultinomialNB

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()

text_tf= tf.fit_transform(data['Phrase'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    text_tf, data['Sentiment'], test_size=0.3, random_state=123)
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)

print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))