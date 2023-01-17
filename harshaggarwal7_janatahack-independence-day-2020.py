import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import bokeh

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

from matplotlib import style

import re

import time

import string

import warnings



# for all NLP related operations on text

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.classify import NaiveBayesClassifier

from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB, MultinomialNB



# To identify the sentiment of text

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

from textblob.np_extractors import ConllExtractor



# ignoring all the warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# downloading stopwords corpus

nltk.download('stopwords')

#nltk.download('wordnet')

#nltk.download('vader_lexicon')

#nltk.download('averaged_perceptron_tagger')

#nltk.download('movie_reviews')

#nltk.download('punkt')

#nltk.download('conll2000')

#nltk.download('brown')

stopwords = set(stopwords.words("english"))



# for showing all the plots inline

%matplotlib inline
train = pd.read_csv("/kaggle/input/topic-modeling-avjh/train.csv")

test = pd.read_csv("/kaggle/input/topic-modeling-avjh/test.csv")

sample_submission = pd.read_csv("/kaggle/input/topic-modeling-avjh/sample_submission.csv")
sample_submission
display("Train File",train.head(10))

display("Test File", test.head(10))
def remove_pattern(text, pattern_regex):

    r = re.findall(pattern_regex, text)

    for i in r:

        text = re.sub(i, '', text)

    

    return text
train['tidy_tweets_title'] = np.vectorize(remove_pattern)(train['TITLE'], "@[\w]*: | *RT*")

train['tidy_tweets_abstract'] = np.vectorize(remove_pattern)(train['ABSTRACT'], "@[\w]*: | *RT*")



test['tidy_tweets_title'] = np.vectorize(remove_pattern)(test['TITLE'], "@[\w]*: | *RT*")

test['tidy_tweets_abstract'] = np.vectorize(remove_pattern)(test['ABSTRACT'], "@[\w]*: | *RT*")
cleaned_tweets_tt = []

cleaned_tweets_ta = []

cleaned_tweets_tet = []

cleaned_tweets_tea = []



for index, row in train.iterrows():

    # Here we are filtering out all the words that contains link

    words_without_links_title = [word for word in row.tidy_tweets_title.split() if 'http' not in word]

    cleaned_tweets_tt.append(' '.join(words_without_links_title))

    

    words_without_links_abstract = [word for word in row.tidy_tweets_abstract.split() if 'http' not in word]

    cleaned_tweets_ta.append(' '.join(words_without_links_abstract))

    

for index, row in test.iterrows():

    words_without_links_title = [word for word in row.tidy_tweets_title.split() if 'http' not in word]

    cleaned_tweets_tet.append(' '.join(words_without_links_title))

    

    words_without_links_abstract = [word for word in row.tidy_tweets_abstract.split() if 'http' not in word]

    cleaned_tweets_tea.append(' '.join(words_without_links_abstract))

    

    

train['tidy_tweets_title'] = cleaned_tweets_tt

train['tidy_tweets_abstract'] = cleaned_tweets_ta

test['tidy_tweets_title'] = cleaned_tweets_tet

test['tidy_tweets_abstract'] = cleaned_tweets_tea
train.shape
train = train[train['tidy_tweets_title']!='']

train = train[train['tidy_tweets_abstract']!='']

test = test[test['tidy_tweets_title']!='']

test = test[test['tidy_tweets_abstract']!='']
'''train.drop_duplicates(subset=['tidy_tweets_title'], keep=False)

train.drop_duplicates(subset=['tidy_tweets_abstract'], keep=False)

test.drop_duplicates(subset=['tidy_tweets_title'], keep=False)

test.drop_duplicates(subset=['tidy_tweets_abstract'], keep=False)'''
train = train.reset_index(drop=True)

test = test.reset_index(drop=True)
train['absolute_tidy_tweets_title'] = train['tidy_tweets_title'].str.replace("[^a-zA-Z# ]", "")

train['absolute_tidy_tweets_abstract'] = train['tidy_tweets_abstract'].str.replace("[^a-zA-Z# ]", "")

test['absolute_tidy_tweets_title'] = test['tidy_tweets_title'].str.replace("[^a-zA-Z# ]", "")

test['absolute_tidy_tweets_abstract'] = test['tidy_tweets_abstract'].str.replace("[^a-zA-Z# ]", "")
stopwords_set = set(stopwords)

cleaned_tweets_tt1 = []

cleaned_tweets_ta1 = []

cleaned_tweets_tet1 = []

cleaned_tweets_tea1 = []





for index, row in train.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords_title = [word for word in row.absolute_tidy_tweets_title.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets_tt1.append(' '.join(words_without_stopwords_title))

    

    # filerting out all the stopwords 

    words_without_stopwords_abstract = [word for word in row.absolute_tidy_tweets_abstract.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets_ta1.append(' '.join(words_without_stopwords_abstract))

    

    

for index, row in test.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords_title = [word for word in row.absolute_tidy_tweets_title.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets_tet1.append(' '.join(words_without_stopwords_title))

    

    # filerting out all the stopwords 

    words_without_stopwords_abstract = [word for word in row.absolute_tidy_tweets_abstract.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets_tea1.append(' '.join(words_without_stopwords_abstract))

    

train['absolute_tidy_tweets_title'] = cleaned_tweets_tt1

train['absolute_tidy_tweets_abstract'] = cleaned_tweets_ta1

test['absolute_tidy_tweets_title'] = cleaned_tweets_tet1

test['absolute_tidy_tweets_abstract'] = cleaned_tweets_tea1    
tokenized_tweet_title = train['absolute_tidy_tweets_title'].apply(lambda x: x.split())

tokenized_tweet_abstract = train['absolute_tidy_tweets_abstract'].apply(lambda x: x.split())



tokenized_tweet_test_title = test['absolute_tidy_tweets_title'].apply(lambda x: x.split())

tokenized_tweet_test_abstract = test['absolute_tidy_tweets_abstract'].apply(lambda x: x.split())
word_lemmatizer = WordNetLemmatizer()



tokenized_tweet_title = tokenized_tweet_title.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet_abstract = tokenized_tweet_abstract.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet_test_title = tokenized_tweet_test_title.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet_test_abstract = tokenized_tweet_test_abstract.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])
for i, tokens in enumerate(tokenized_tweet_title):

    tokenized_tweet_title[i] = ' '.join(tokens)



for i, tokens in enumerate(tokenized_tweet_abstract):

    tokenized_tweet_abstract[i] = ' '.join(tokens)



for i, tokens in enumerate(tokenized_tweet_test_title):

    tokenized_tweet_test_title[i] = ' '.join(tokens)

    

for i, tokens in enumerate(tokenized_tweet_test_abstract):

    tokenized_tweet_test_abstract[i] = ' '.join(tokens)





train['absolute_tidy_tweets_title'] = tokenized_tweet_title

train['absolute_tidy_tweets_abstract'] = tokenized_tweet_abstract

test['absolute_tidy_tweets_title'] = tokenized_tweet_test_title

test['absolute_tidy_tweets_abstract'] = tokenized_tweet_test_abstract
textblob_key_phrases_title = []

textblob_key_phrases_abstract = []

textblob_key_phrases_test_title = []

textblob_key_phrases_test_abstract = []

extractor = ConllExtractor()



for index, row in train.iterrows():

    # filerting out all the hashtags

    words_without_hash_title = [word for word in row.tidy_tweets_title.split() if '#' not in word.lower()]

    

    hash_removed_sentence_title = ' '.join(words_without_hash_title)

    

    blob = TextBlob(hash_removed_sentence_title, np_extractor=extractor)

    textblob_key_phrases_title.append(list(blob.noun_phrases))

    

    words_without_hash_abstract = [word for word in row.tidy_tweets_abstract.split() if '#' not in word.lower()]

    

    hash_removed_sentence_abstract = ' '.join(words_without_hash_abstract)

    

    blob = TextBlob(hash_removed_sentence_abstract, np_extractor=extractor)

    textblob_key_phrases_abstract.append(list(blob.noun_phrases))



for index, row in test.iterrows():

    # filerting out all the hashtags

    words_without_hash_title = [word for word in row.tidy_tweets_title.split() if '#' not in word.lower()]

    

    hash_removed_sentence_title = ' '.join(words_without_hash_title)

    

    blob = TextBlob(hash_removed_sentence_title, np_extractor=extractor)

    textblob_key_phrases_test_title.append(list(blob.noun_phrases))

    

    words_without_hash_abstract = [word for word in row.tidy_tweets_abstract.split() if '#' not in word.lower()]

    

    hash_removed_sentence_abstract = ' '.join(words_without_hash_abstract)

    

    blob = TextBlob(hash_removed_sentence_abstract, np_extractor=extractor)

    textblob_key_phrases_test_abstract.append(list(blob.noun_phrases))
train['key_phrases_title'] = textblob_key_phrases_title

train['key_phrases_abstract'] = textblob_key_phrases_abstract



test['key_phrases_title'] = textblob_key_phrases_test_title

test['key_phrases_abstract'] = textblob_key_phrases_test_abstract
display("Train File",train.head(10))

display("Test File", test.head(10))
train.drop(['tidy_tweets_title','tidy_tweets_abstract','absolute_tidy_tweets_title','absolute_tidy_tweets_abstract'],axis=1,inplace=True)

test.drop(['tidy_tweets_title','tidy_tweets_abstract','absolute_tidy_tweets_title','absolute_tidy_tweets_abstract'],axis=1,inplace=True)
train['KEY_PHRASES'] = train['key_phrases_title'] + train['key_phrases_abstract']

test['KEY_PHRASES'] = test['key_phrases_title'] + test['key_phrases_abstract']
train = train[['ID', 'TITLE', 'ABSTRACT', 'key_phrases_title', 'key_phrases_abstract','KEY_PHRASES','Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']]
display("Train File",train.head(10))

display("Test File", test.head(10))
train.drop(['key_phrases_title','key_phrases_abstract'],axis=1, inplace=True)

test.drop(['key_phrases_title','key_phrases_abstract'],axis=1, inplace=True)
display(train.columns)

display(test.columns)
display("Train File",train.head(10))

display("Test File", test.head(10))
train['KEY_PHRASES'] = train['KEY_PHRASES'].apply(lambda x: ' '.join(x))
test['KEY_PHRASES'] = test['KEY_PHRASES'].apply(lambda x: ' '.join(x))
tfidf_word_vectorizer = TfidfVectorizer()

# TF-IDF feature matrix

X = tfidf_word_vectorizer.fit_transform(train['KEY_PHRASES']).toarray()
# TF-IDF feature matrix

X_test = tfidf_word_vectorizer.transform(test['KEY_PHRASES']).toarray()
y = train.iloc[:,4:10]

y = y.astype(str)
train['Condition']=y.values.tolist()
def listToString(s):

    str1 = " "  

    return (str1.join(s))



train['Dependent']=train['Condition'].apply(lambda x:listToString(x))

y=train['Dependent']
naive_classifier = MultinomialNB()

naive_classifier.fit(X, y)



# predictions over test set

predictions = naive_classifier.predict(X_test)
predictions
test['Predictions'] = predictions
test[['Computer Science ','Physics' ,'Mathematics' ,'Statistics','Quantitative Biology' ,'Quantitative Finance']] = test.Predictions.str.split(" ",expand=True)
test
submission = test.drop(["TITLE","ABSTRACT","KEY_PHRASES","Predictions"],axis=1)
submission.to_csv('submission2.csv')
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=7000,shuffle=True)

mlp.fit(X, y)
mlp_predictions = mlp.predict(X_test)
test.columns
mlp_test1 = test.drop(['Predictions',

       'Computer Science ', 'Physics', 'Mathematics', 'Statistics',

       'Quantitative Biology', 'Quantitative Finance'], axis=1)
mlp_test1['Predictions'] = mlp_predictions
mlp_test1[['Computer Science ','Physics' ,'Mathematics' ,'Statistics','Quantitative Biology' ,'Quantitative Finance']] = mlp_test1.Predictions.str.split(" ",expand=True)
submission = mlp_test1.drop(["TITLE","ABSTRACT","KEY_PHRASES","Predictions"],axis=1)

submission.to_csv('submission_mlp.csv')