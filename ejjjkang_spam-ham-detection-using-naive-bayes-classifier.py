# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier



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
message = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', usecols=['v1', 'v2'],encoding='latin-1')

message.head()
message = message.rename({'v1': 'label', 'v2': 'message'}, axis=1) 
message.describe()
message['length'] = message['message'].apply(len)

message.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
message['length'].plot(bins=50,kind='hist')

message.describe()
message[message["length"]==910]['message'].iloc[0]
import string



test_message = "To, be or not to be that is the question~~"

pre_message = [char for char in test_message if char not in string.punctuation]

pre_message = ''.join(pre_message)

print(pre_message)
from nltk.corpus import stopwords

stopwords.words('english')[0:10]
pre_message.split()
clean_mess = [ word for word in pre_message.split() if word.lower() not in stopwords.words('english')]
clean_mess
def text_Process(text):

    pre_message = [char for char in text if char not in string.punctuation]

    pre_message = ''.join(pre_message)

    clean_mess = [ word for word in pre_message.split() if word.lower() not in stopwords.words('english')]

    return clean_mess

    
message['message'].head(5).apply(text_Process)
from sklearn.feature_extraction.text import CountVectorizer



bow_transformer = CountVectorizer(analyzer=text_Process).fit(message['message'])

print(len(bow_transformer.vocabulary_))
message4=message['message'][2]

print(message4)
bow4 = bow_transformer.transform([message4])

print(bow4)
bow = bow_transformer.transform(message['message'])
print('Shape of Sparse Matrix: ',bow.shape)

print('Amount of non-zero occurences:',bow.nnz)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer().fit(bow)
messages_tfidf=tfidf_transformer.transform(bow)

print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf,message['label'])

tfidf4 = tfidf_transformer.transform(bow4)
print('predicted:',spam_detect_model.predict(tfidf4)[0])

print('expected:',message.label[3])

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(message['label'],all_predictions))

print(confusion_matrix(message['label'],all_predictions))
all_predictions = spam_detect_model.predict(messages_tfidf)

print(all_predictions)

from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(message['message'],message['label'],test_size=0.2)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([

   ( 'bow',CountVectorizer(analyzer=text_Process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB()),

])
pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
