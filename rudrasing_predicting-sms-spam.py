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
import pandas as pd

spam = pd.read_csv(r"../input/sms-spam-collection-dataset/spam.csv",encoding = 'latin1')
spam = spam[['v1','v2']]
spam.columns = ['labels','messages']
spam.groupby('labels').describe()
spam['length'] = spam['messages'].apply(len)
spam['length'].plot(kind = 'hist',bins = 50)
spam['length'].describe()
spam[spam['length'] == 910]['messages']
spam.hist(column = 'length',by = 'labels',bins = 50)
import string

import re

string.punctuation
spam['messages'] = spam['messages'].apply(lambda x : ''.join([char for char in x if char not in string.punctuation]))

#spam['messages'] = spam['messages'].apply(lambda x : )
from nltk.corpus import stopwords

spam['messages'] = spam['messages'].apply(lambda x : ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer()

bow_transformer.fit(spam['messages'])
mess4 = spam['messages'][3]

mess4
print(bow_transformer.transform([mess4]))
bow_transformer.get_feature_names()[7196]
ms5 = [spam['messages'][0]]

bow5 = bow_transformer.transform(ms5)
print(bow_transformer.transform(ms5))
messages_bow = bow_transformer.transform(spam['messages'])
messages_bow.shape
messages_bow.nnz # amount of non zero occurances
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer  = TfidfTransformer()

tfidf_transformer.fit(messages_bow)
print(tfidf_transformer.transform(bow5))
tfidf_transformer.idf_[bow_transformer.vocabulary_['go']]
bow_transformer.vocabulary_['go']
tfidf_transformer.idf_[bow_transformer.vocabulary_['what']] # number to feature dictonary
messages_tfidf = tfidf_transformer.transform(messages_bow)
messages_tfidf.shape
messages_tfidf.nnz
from sklearn.naive_bayes import MultinomialNB
naive_bays = MultinomialNB()

spam_detect_model = naive_bays.fit(messages_tfidf,spam['labels'])
message = [spam['messages'][56]]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer.transform(bow_transformer.transform(message))
spam_detect_model.predict(tfidf_transformer.transform(bow_transformer.transform(message)))
spam['messages'][666]
[spam['messages'][56]]
spam_detect_model.predict(tfidf_transformer.transform(bow_transformer.transform([spam['messages'][56]])))
pred = spam_detect_model.predict(messages_tfidf)
from sklearn.metrics import classification_report
print(classification_report(pred,spam['labels']))
spam
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
X_train,X_test,y_train,y_test = train_test_split(spam['messages'],spam['labels'],test_size = 0.2)
#clf = Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),('naive_bays',MultinomialNB())])

clf = Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),('nbclass',MultinomialNB())],verbose = True)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(classification_report(pred,y_test))