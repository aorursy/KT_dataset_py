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
message= pd.read_csv("../input/spam.csv", encoding='ISO-8859-1')
message.head()

message.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis= 1,inplace=True)
message.rename(columns={'v1':'label','v2':'messages'},inplace=True)
message.head()
message['length'] = message['messages'].apply(len)
message.head()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
message['length'].plot(bins=50, kind = 'hist')
message.hist(columns='length',by='label', bins= 50 )
from nltk.corpus import stopwords
import string
stopwords.words('english')[:10]
def text_preprocess(mess):
    nonpuc = [char for char in mess if char not in string.punctuation]
    nonpuc = ''.join(nonpuc)
    
    return [word for word in nonpuc.split() if word.lower not in stopwords.words('english')]
text_preprocess("my name is it has  Mayur.")
message['messages'].head(5).apply(text_preprocess)
from sklearn.feature_extraction.text import CountVectorizer
bow_trans = CountVectorizer(analyzer=text_preprocess).fit(message['messages'])
print(len(bow_trans.vocabulary_))
messages_bow = bow_trans.transform(message['messages'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, message['label'])
from sklearn.metrics import classification_report
all_predictions = spam_detect_model.predict(messages_tfidf)
print (classification_report(message['label'], all_predictions))
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(message['messages'], message['label'], test_size=0.3)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_preprocess)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))