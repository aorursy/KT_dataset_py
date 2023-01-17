# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from math import log, sqrt

import pandas as pd

import numpy as np

import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
messages = pd.read_csv("../input/spam.csv",encoding="latin-1")

messages.head()
#Drop the unnamed columns

messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1,inplace=True)

messages.head()
messages = messages.rename(columns={"v1":"label","v2":"text"})

messages.head()
messages['label'].value_counts()

messages['label'] = messages['label'].map({'ham':0,'spam':1})

messages.head()
X_train,X_test,y_train,y_test = train_test_split(messages["text"],messages["label"], test_size = 0.2, random_state = 10)

v = CountVectorizer()

v.fit(X_train)  #build vocabulary from the messages in the data

#print(v.vocabulary_)
train_df = v.transform(X_train)

test_df = v.transform(X_test)

hamwords = ''

spamwords = ''

hamw = messages[messages['label']==0]['text']

spamw = messages[messages['label']==1]['text']

for row in hamw:

    words = word_tokenize(row)

    #print(word)

    for x in words:

        hamwords += x + ' '

for row in spamw:

    words = word_tokenize(row)

    #print(word)

    for x in words:

        spamwords += x + ' '
wc_spam = WordCloud().generate(spamwords)

wc_ham = WordCloud().generate(hamwords)

#Spam Word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(wc_spam)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()

#Ham word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(wc_ham)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#Logistic Regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(train_df,y_train)
predictions= model.predict(test_df)

accuracy_score(y_test,predictions)