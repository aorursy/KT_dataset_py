import numpy as np
import pandas as pd
import os
import string
import re

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict 

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
ds_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
ds_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(ds_train.shape)
print(ds_test.shape)
ds_train.head(5)
#Counts of missing keywords
ds_train.isnull()['keyword'].value_counts()
#plot count of missing keywords
sns.countplot(ds_train.isnull()['keyword'])
# Counts of missing locations
ds_train.isnull()['location'].value_counts()
#plot count of missing locations
sns.countplot(ds_train.isnull()['location'] )
count = ds_train['target'].value_counts()
percentage = ds_train['target'].value_counts(normalize= True)*100
pd.concat([count, percentage],  axis=1, keys = ['count', '%'])
sns.countplot(ds_train['target'], hue=ds_train['target'])
# Counts of disaster keywords
disaster_keywords = ds_train[ds_train['target']==1]['keyword'].value_counts()
sns.barplot(x= disaster_keywords[0:20], y =disaster_keywords.index[:20])

# Counts of non-disaster keywords
non_disaster_keywords = ds_train[ds_train['target']==0]['keyword'].value_counts()
sns.barplot(x= non_disaster_keywords[0:20], y = non_disaster_keywords.index[:20])
def hash_words(text):
    '''
       Return a words which start with "#"
    '''
    return ', '.join(re.findall(r'(?<=#)\w+', text))
print(ds_train['text'][4])
print(hash_words(ds_train['text'][4]))
# apply len method to get length of tweet text
ds_train['text_len'] = ds_train['text'].apply(len)
ds_train['words_count'] = ds_train['text'].apply(lambda x: len(str(x).split()))
ds_train.head(5)

#Difference among all tokenizer, provided by nltk library
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
#applying cleaning process

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[{}]'.format(re.escape(string.punctuation)), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
ds_train['clean_text'] = ds_train['text'].apply(clean_text)
ds_train.head(5)
# remove stopwords

def remove_stopwords(text):
    '''
    remove english stop words
    '''
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])
#remove stop words
ds_train['clean_text'] = ds_train['clean_text'].apply(lambda x : remove_stopwords(x))
ds_train.head(5)
# Creating a corpus
corpus = [txt for txt in ds_train['clean_text']]
#unique words
unique_words = Counter(' '.join(corpus).split())
unique_words = pd.DataFrame.from_dict(unique_words, orient='index',columns=['WordFrequency'])
unique_words.sort_values(by=['WordFrequency'], inplace = True, ascending = False)
unique_words[:20]
#Keep only those unique words which are having frequecy > 15
unique_words = unique_words[unique_words['WordFrequency'] > 15]
len(unique_wordsque_words)
# Creating the Bag of Words model

tf_vec = TfidfVectorizer(max_features=len(unique_words), ngram_range=(1,2))
X = tf_vec.fit_transform(corpus).todense()
y = ds_train['target'].values
#Splitting data into training and validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
cls_lr = LogisticRegression()
cls_lr.fit(X_train, y_train)
y_pred = cls_lr.predict(X_test)
confusion_matrix(y_test, y_pred)
#Calculating Model Accuracy
print('Logistic Regression Model Accuracy Score for Train Data set is {}'.format(cls_lr.score(X_train, y_train)))
print('Logistic Regression Model Accuracy Score for Test Data set is {}'.format(cls_lr.score(X_test, y_test)))
print('Logistic Regression Model F1 Score is {}'.format(f1_score(y_test, y_pred)))

classfier_nb = MultinomialNB()
classfier_nb.fit(X_train, y_train)
y_pred = classfier_nb.predict(X_test)
confusion_matrix(y_test, y_pred)
#Calculating Model Accuracy
print('Naive bayes Model Accuracy Score for Train Data set is {}'.format(classfier_nb.score(X_train, y_train)))
print('Naive bayes Model Accuracy Score for Test Data set is {}'.format(classfier_nb.score(X_test, y_test)))
print('Naive bayes Model F1 Score is {}'.format(f1_score(y_test, y_pred)))
ds_test.head(5)
ds_test['clean_text'] = ds_test['text'].apply(clean_text)
ds_test['clean_text'] = ds_test['clean_text'].apply(lambda x : remove_stopwords(x))
#transform test dataset using tf idf vectorizer
X_test_set=tf_vec.transform(ds_test['text']).todense()
y_test_pred = classfier_nb.predict(X_test_set)
submission_df=pd.DataFrame({"id":ds_test['id'],"target":y_test_pred.ravel()})
submission_df.to_csv("submission.csv",index=False)