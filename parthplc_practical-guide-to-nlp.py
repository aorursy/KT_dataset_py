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
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

train_df.head()
def get_description(df):

    print(df.info())

    print("*"* 40)

    print(df.describe())

    print("*"* 40)

    print(df.head(10))
get_description(train_df)
get_description(test_df)
import seaborn as sns

import matplotlib.pyplot as plt

x=train_df.target.value_counts()

sns.barplot(x.index,x)

def no_of_characters_graphs(df):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

    tweet_len=df[df['target']==1]['text'].str.len()

    ax1.hist(tweet_len,color='red')

    ax1.set_title('Disaster tweets')

    tweet_len=df[df['target']==0]['text'].str.len()

    ax2.hist(tweet_len,color='green')

    ax2.set_title('Not disaster tweets')

    fig.suptitle('Characters in tweets')

    plt.show()
no_of_characters_graphs(train_df)
def no_words_graphs(df):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

    tweet_len=df[df['target']==1]['text'].str.split().map(lambda x: len(x))

    ax1.hist(tweet_len,color='red')

    ax1.set_title('disaster tweets')

    tweet_len=df[df['target']==0]['text'].str.split().map(lambda x: len(x))

    ax2.hist(tweet_len,color='green')

    ax2.set_title('Not disaster tweets')

    fig.suptitle('Words in a tweet')

    plt.show()
no_words_graphs(train_df)
from nltk.corpus import stopwords



stop=set(stopwords.words('english'))

from collections import defaultdict

from collections import  Counter
def create_corpus(target):

    corpus=[]

    

    for x in train_df[train_df['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
# Common stopwords in tweets
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

x,y=zip(*top)

plt.bar(x,y)
corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x,y=zip(*top)

plt.bar(x,y)
import re
message ="my number is 510-123-4567"

myregex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

match = myregex.search(message)

print(match.group())
import re

txt = "my number is 990-445-4836 and 844-096-1968"

myregex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

print(myregex.findall(txt))



            
# Basic EDA
train_df.isnull().sum()
#Missing values in test set

test_df.isnull().sum()
train_df['target'].value_counts()
sns.barplot(train_df['target'].value_counts().index,train_df['target'].value_counts(),palette='rocket')
# A disaster tweet

disaster_tweets = train_df[train_df['target']==1]['text']

disaster_tweets.values[1]
non_disaster_tweets = train_df[train_df['target']==0]['text']

non_disaster_tweets
train_df['keyword'].value_counts()[:10]
test_df['keyword'].value_counts()[:10]
# Applying a first round of text cleaning techniques
# Applying a first round of text cleaning techniques

import string

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))

test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train_df['text'].head()
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disaster Tweets',fontsize=40);
import nltk

text = "Are you coming , aren't you"

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')



print("Example Text: ",text)

print("------------------------------------------------------------------------------------------------")

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train_df['text'] = train_df['text'].apply(lambda x: tokenizer.tokenize(x))

test_df['text'] = test_df['text'].apply(lambda x: tokenizer.tokenize(x))

train_df['text'].head()
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train_df['text'] = train_df['text'].apply(lambda x : remove_stopwords(x))

test_df['text'] = test_df['text'].apply(lambda x : remove_stopwords(x))

train_df.head(10)
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train_df['text'] = train_df['text'].apply(lambda x : combine_text(x))

test_df['text'] = test_df['text'].apply(lambda x : combine_text(x))

train_df['text']

train_df.head()
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df['text'])

test_vectors = count_vectorizer.transform(test_df["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors.shape)
sentences = ['The weather is sunny', 'The weather is partly sunny and partly cloudy.']
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(sentences)

vectorizer.vocabulary_
vectorizer.transform(sentences).toarray()

# StopWord removal using countvectorizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')



count_vectorizer = CountVectorizer(stop_words = stopwords)

count_vectorizer.fit(train_df['text'])



train_vectors = count_vectorizer.fit_transform(train_df['text'])

test_vectors = count_vectorizer.transform(test_df["text"])



## Keeping only non-zero elements to preserve space 

train_vectors.shape
train_df.head()
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df['text'])

test_vectors = count_vectorizer.transform(test_df["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())

train_df
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
# Submission

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False)
# Well I got a score of .078527


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# word level

tfidf = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}',max_features=5000)

train_tfidf = tfidf.fit_transform(train_df['text'])

test_tfidf = tfidf.transform(test_df["text"])
tfidf_vectorizer = TfidfVectorizer( min_df=3,  max_features=None,analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = stopwords)



train_tfidf = tfidf.fit_transform(train_df['text'])

test_tfidf = tfidf.transform(test_df["text"])
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_tfidf, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_tfidf, train_df["target"])

# Submission

sample_submission_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission_1["target"] = clf.predict(test_tfidf)

sample_submission_1.to_csv("submission_1.csv", index=False)
# Naives Bayes Clssifiers

from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
# Fitting a simple Naive Bayes on Counts

clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_tfidf, train_df["target"], cv=5, scoring="f1")

scores
clf_NB.fit(train_tfidf, train_df["target"])

sample_submission_2 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission_2["target"] = clf_NB.predict(test_tfidf)

sample_submission_2.to_csv("submission_2.csv", index=False)