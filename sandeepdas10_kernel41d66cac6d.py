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
msg=[line.rstrip() for line  in open('../input/smsspamcollection/SMSSpamCollection')]

print(len(msg))
for msg_no,msg in enumerate(msg[:8]):

    print(msg_no,msg)

    print('\n')
import pandas as pd
msg=pd.read_csv('../input/smsspamcollection/SMSSpamCollection',sep='\t',names=["Label","Message"])

msg.head()
msg.describe()



msg.groupby('Label').describe()
msg['Length']=msg['Message'].apply(len)

msg.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
msg['Length'].plot(bins=50,kind='hist',color='Red')
msg.Length.describe()
msg[msg['Length']==910]['Message'].iloc[0]
import string

mess = 'This is my text message!...'

nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

print(nopunc)
from nltk.corpus import stopwords

stopwords.words('english')[0:10]

nopunc.split()
mod_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

mod_mess
def text_process(mess):

    nopunc =[char for char in mess if char not in string.punctuation]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
msg.head()
msg['Message'].head(10).apply(text_process)

msg.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_transf = CountVectorizer(analyzer=text_process).fit(msg['Message'])

print(len(bow_transf.vocabulary_))
msg6=msg['Message'][5]

print(msg6)
bow6=bow_transf.transform([msg6])

print(bow6)

print(bow6.shape)
print(bow_transf.get_feature_names()[4893])

print(bow_transf.get_feature_names()[9641])
messages_bow = bow_transf.transform(msg['Message'])
print('Shape of Sparse Matrix: ',messages_bow.shape)

print('Amount of non-zero occurences:',messages_bow.nnz)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transf=TfidfTransformer().fit(messages_bow)

tfidf6 = tfidf_transf.transform(bow6)

print(tfidf6)
print(tfidf_transf.idf_[bow_transf.vocabulary_['n']])

print(tfidf_transf.idf_[bow_transf.vocabulary_['natural']])
messages_tfidf=tfidf_transf.transform(messages_bow)

print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf,msg['Label'])
print('Predicted:',spam_detect_model.predict(tfidf6)[0])

print('Actual:',msg.Label[4])

all_predictions = spam_detect_model.predict(messages_tfidf)

print(all_predictions)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(msg['Label'],all_predictions))

print(confusion_matrix(msg['Label'],all_predictions))
from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(msg['Message'],msg['Label'],test_size=0.3)
print(len(msg_train),len(msg_test),len(label_train),len(label_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([

   ( 'bow',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB()),

])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))