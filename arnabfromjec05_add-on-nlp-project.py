# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import nltk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1 = open('../input/amazon_cells_labelled.txt')
text=data1.read()
import re

sent=re.findall(r'.*\n',text)

sent
sent=[(line[:-3],int(line[-2])) for line in sent]

sent
data=pd.DataFrame(sent,columns=['review','score'])
data
print(sum(data['score'])) #equally mixed data
import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

ps=PorterStemmer()

def pre_process(text):

    clean_text=[char for char in text if char not in string.punctuation]

    clean_text="".join(clean_text)

    clean_text=[words.lower() for words in clean_text.split()]

    clean_text=[words for words in clean_text if words not in stopwords.words('english')]

#     clean_text=[ps.stem(words) for words in clean_text]

    return clean_text

#     send=[]

#     for i in range(len(clean_text)):

#         if i+1<=len(clean_text)-1:

#             send.append((clean_text[i],clean_text[i+1]));

#     return send



pre_process("Hello i am a singer and singing")
from sklearn.feature_extraction.text import CountVectorizer

x = CountVectorizer(analyzer=pre_process).fit(data['review'])
len(x.vocabulary_)
x = x.transform(data['review'])

print(x)
print(x.shape)
y=data['score']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(x_train,y_train)
pred=nb.predict(x_test)
from sklearn.metrics import accuracy_score

print("ACCURACY : "+str(accuracy_score(y_test,pred)))
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print('\n')

print(classification_report(y_test, pred))
from sklearn import svm

svmc=svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

svmc.fit(x_train,y_train)
pred1=svmc.predict(x_test)

print("ACCURACY : "+str(accuracy_score(y_test,pred1)))
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred1))

print('\n')

print(classification_report(y_test, pred1))
test="I don't like this charger"

x = CountVectorizer(analyzer=pre_process).fit(data['review'])

test=x.transform([test])

nb.predict(test)[0]
test='Waste product and failure woks'

x = CountVectorizer(analyzer=pre_process).fit(data['review'])

test=x.transform([test])

svmc.predict(test)[0]
test='Belt is not good quality... not worth for money'

x = CountVectorizer(analyzer=pre_process).fit(data['review'])

test=x.transform([test])

svmc.predict(test)[0]