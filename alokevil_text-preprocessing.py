import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk import ngrams

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
input_str = "John Wick (retrospectively known as John Wick: Chapter 1) is a 2014 American neo-noir action thriller film directed by Chad Stahelski and written by Derek Kolstad. It stars Keanu Reeves, Michael Nyqvist, Alfie Allen, Adrianne Palicki, Bridget Moynahan, Dean Winters, Ian McShane, John Leguizamo, and Willem Dafoe.It grossed $88 million worldwide against a production budget of $20 million."

input_str
lower_str = input_str.lower()

lower_str
punc_free = input_str.translate(str.maketrans('', '', string.punctuation))

punc_free
# setting stop words for English language

stop_words = set(stopwords.words('english'))



tokens = word_tokenize(punc_free)

result = [i for i in tokens if not i in stop_words]

result
stemmer = PorterStemmer()

stem_str = word_tokenize(punc_free)

for word in stem_str:

    print(stemmer.stem(word))
lemmatizer = WordNetLemmatizer()

lemm_str = word_tokenize(punc_free)

for word in lemm_str:

    print(lemmatizer.lemmatize(word))
tokens = word_tokenize(input_str)

tokens
n = 3

three_grams = ngrams(lower_str.split(), n)



for grams in three_grams:

    print(grams)
docs =["john wick (retrospectively known as john wick: chapter 1) is a 2014 american neo-noir action thriller film directed by chad stahelski and written by derek kolstad",

       "it stars keanu reeves, michael nyqvist, alfie allen, adrianne palicki, bridget moynahan, dean winters, ian mcshane, john leguizamo, and willem dafoe" ,

       "it grossed $88 million worldwide against a production budget of $20 million"]

vec = CountVectorizer()

X = vec.fit_transform(docs)



df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df
docs =["john wick (retrospectively known as john wick: chapter 1) is a 2014 american neo-noir action thriller film directed by chad stahelski and written by derek kolstad",

       "it stars keanu reeves, michael nyqvist, alfie allen, adrianne palicki, bridget moynahan, dean winters, ian mcshane, john leguizamo, and willem dafoe" ,

       "it grossed $88 million worldwide against a production budget of $20 million"]



vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(docs)



df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

df