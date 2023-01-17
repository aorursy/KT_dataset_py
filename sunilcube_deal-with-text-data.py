'''Contents

    Basic feature extraction using text data

        Number of words

        Number of characters

        Average word length

        Number of stopwords

        Number of special characters

        Number of numerics

        Number of uppercase words

    Basic Text Pre-processing of text data

        Lower casing

        Punctuation removal

        Stopwords removal

        Frequent words removal

        Rare words removal

        Spelling correction

    Tokenization

        Stemming

        Lemmatization

    Advance Text Processing

        N-grams

        Term Frequency

        Inverse Document Frequency

        Term Frequency-Inverse Document Frequency (TF-IDF)

        Bag of Words

        Hashing with HashingVectorizer

'''
import os

import pandas as pd

from nltk.corpus import stopwords

from textblob import TextBlob

import numpy as np
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
train_df.head(5)
#Number of Words



train_df['word_count'] = train_df['question_text'].apply(lambda x: len(str(x).split(" ")))

train_df[['question_text','word_count']].head()

#Number of characters



train_df['char_count'] = train_df['question_text'].str.len() ## this also includes spaces

train_df[['question_text','char_count']].head()
#Average Word Length



def avg_word(sentence):

  words = sentence.split()

  return (sum(len(word) for word in words)/len(words))



train_df['avg_word'] = train_df['question_text'].apply(lambda x: avg_word(x))

train_df[['question_text','avg_word']].head()
#Number of stopwords



stop = stopwords.words('english')

train_df['stopwords'] = train_df['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))

train_df[['question_text','stopwords']].head()

#Number of special characters



train_df['hastags'] = train_df['question_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

train_df[['question_text','hastags']].head()
#Number of numerics



train_df['numerics'] = train_df['question_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

train_df[['question_text','numerics']].head()
# Number of Uppercase words



train_df['upper'] = train_df['question_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

train_df[['question_text','upper']].head()
#Lower case



train_df['question_text'] = train_df['question_text'].apply(lambda x: x.lower())

test_df['question_text'] = test_df['question_text'].apply(lambda x: x.lower())
#Removing Punctuation



train_df['question_text'] = train_df['question_text'].str.replace('[^\w\s]','')

train_df['question_text'].head()

#Removal of Stop Words



train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train_df['question_text'].head()
#Common word removal



freq = pd.Series(' '.join(train_df['question_text']).split()).value_counts()[:10]

freq = list(freq.index)

train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

train_df['question_text'].head()
#Rare words removal



freq = pd.Series(' '.join(train_df['question_text']).split()).value_counts()[-10:]

freq = list(freq.index)

train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

train_df['question_text'].head()
#Spelling correction

# train_df['question_text'] = train_df['question_text'].apply(lambda x: str(TextBlob(x).correct()))

# train_df['question_text'].head()



train_df['question_text'][:10].apply(lambda x: str(TextBlob(x).correct()))

train_df['question_text'].head()
#Tokenization

'''Tokenization refers to dividing the text into a sequence of words or sentences.'''



TextBlob(train_df['question_text'][1]).words

#Stemming

'''Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. 

For this purpose, we will use PorterStemmer from the NLTK library.

'''



from nltk.stem import PorterStemmer

st = PorterStemmer()

train_df['question_text'][:10].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Lemmatization

'''Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices.

'''



from textblob import Word

train_df['question_text'] = train_df['question_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train_df['question_text'].head()
#N-grams

'''N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. 

Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used. '''



TextBlob(train_df['question_text'][0]).ngrams(2)
#Term frequency

'''Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.

TF = (Number of times term T appears in the particular row) / (number of terms in that row)'''



tf1 = (train_df['question_text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1.columns = ['words','tf']

tf1



#Inverse Document Frequency

'''

The intuition behind inverse document frequency (IDF) is that a word is not of much use to us if it’s appearing in all the documents.

The IDF of each word is the log of the ratio of the total number of rows to the number of rows in which that word is present.

IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.

'''



for i,word in enumerate(tf1['words']):

  tf1.loc[i, 'idf'] = np.log(train_df.shape[0]/(len(train_df[train_df['question_text'].str.contains(word)])))



tf1
#Term Frequency – Inverse Document Frequency (TF-IDF)

tf1['tfidf'] = tf1['tf'] * tf1['idf']

tf1
#Using sklearn has a separate function to directly obtain TF and IDF 



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',

 stop_words= 'english',ngram_range=(1,1))

train_vect = tfidf.fit_transform(train_df['question_text'])

#Bag of Words

'''

Bag of Words (BoW) refers to the representation of text which describes the presence of words within the text data. The intuition behind this is that two similar text fields will contain similar kind of words, and will therefore have a similar bag of words.

'''

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

train_bow = bow.fit_transform(train_df['question_text'])

#Hashing with HashingVectorizer

'''Counts and frequencies can be very useful, but one limitation of these methods is that the vocabulary can become very large.

This, in turn, will require large vectors for encoding documents and impose large requirements on memory and slow down algorithms.

A clever work around is to use a one way hash of words to convert them to integers. The clever part is that no vocabulary is required and you can choose an arbitrary-long fixed length vector. A downside is that the hash is a one-way function so there is no way to convert the encoding back to a word (which may not matter for many supervised learning tasks).

'''



from sklearn.feature_extraction.text import HashingVectorizer

# create the transform

vectorizer = HashingVectorizer(n_features=100)

vector = vectorizer.fit_transform(train_df['question_text'])