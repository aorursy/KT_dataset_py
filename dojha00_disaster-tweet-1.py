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
import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
stop=set(stopwords.words('english'))
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def clean_text(text):

    text=text.lower()

    text=re.sub('https?://\w+(.)\w+(.)\w+',' ',text)

    text=re.sub('[!@#$%^&*:0-9]+','',text)

    text=re.sub('[()]+',' ',text)

    text=re.sub('\w*\d+\w*','',text)

    text=re.sub('\n','',text)

    return text
# Clean the texts

# train['text']=train['text'].apply(lambda x: clean_text(x))
#Tokenizations

text="Are you n'''t , coming home."

tok1=nltk.tokenize.WhitespaceTokenizer()

tok2=nltk.tokenize.TreebankWordTokenizer()

tok3=nltk.tokenize.WordPunctTokenizer()

tok4=nltk.tokenize.RegexpTokenizer(r'\w+')

print("Whitepsace: ",tok1.tokenize(text));

print("treebank: ",tok2.tokenize(text));

print("Word punc: ",tok3.tokenize(text));

print("Regexp: ",tok4.tokenize(text));
# tok4=nltk.tokenize.RegexpTokenizer(r'\w+')

# train['text']=train['text'].apply(lambda x: tok4.tokenize(x))
#stopwords removal

def clean_stopwords(text):

    word=[w for w in text if w not in stopwords.words('english')]

    return word
# train['text']=train['text'].apply(lambda x: clean_stopwords(x))
# #Stemming Lemmatization

# text='feet cats dogs barked in night very loudly'

# tokenizer=nltk.tokenize.TreebankWordTokenizer()

# tokens=tokenizer.tokenize(text)

# #Stemming

# stemmer=nltk.stem.PorterStemmer()

# words=[stemmer.stem(token) for token in tokens]

# print(words)

# #Lemmatizer

# lem=nltk.stem.WordNetLemmatizer()

# print("  ".join(lem.lemmatize(token) for token in tokens))
#combine text

def combine_words(text_list):

    combined_text=' '.join(text_list)

    return combined_text
# train['text']=train['text'].apply( lambda x :combine_words(x) )
def text_preprocessing(text):

    tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')

    lem=nltk.stem.WordNetLemmatizer()

    text=clean_text(text)    

    text=tokenizer.tokenize(text)

    #text=clean_stopwords(text)

    text=[lem.lemmatize(word) for word in text]

    combined_text=' '.join(text)

    return combined_text
# train['text']=train['text'].apply(lambda x:text_preprocessing(x) )

# test['text']=test['text'].apply(lambda x:text_preprocessing(x) )
count_vectorizer=CountVectorizer()

combine_text=pd.concat([train['text'],test['text']],axis=0)

combine_vectors=count_vectorizer.fit_transform(combine_text)

train_vectors=combine_vectors[0:len(train['text'])]

test_vectors=combine_vectors[len(train['text']):]

#test_vectors=count_vectorizer.fit_transform(test['text'])

#print(train_vectors[0].todense())
tfidf=TfidfVectorizer(min_df=2,max_df=0.5,ngram_range=(1,2))

train_tfidf=tfidf.fit_transform(train['text'])

print(train_tfidf[0].todense())
# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

scores

# with removing stopwords array([0.60623782, 0.53592561, 0.59367397, 0.5271028 , 0.71506635])
# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

scores

# with removing stopwords array([0.55429162, 0.5105215 , 0.56378234, 0.42978723, 0.68280035])
# Fitting a simple Naive Bayes on TFIDF

CountV_NB = MultinomialNB()

scores = cross_val_score(CountV_NB, train_vectors, train["target"], cv=5, scoring="f1")

scores
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
CountV_NB.fit(train_vectors, train["target"])

y=CountV_NB.predict(test_vectors)
result=pd.DataFrame()

result['Id']=test['id']

result['target']=y
result.to_csv('submission.csv',index=False)