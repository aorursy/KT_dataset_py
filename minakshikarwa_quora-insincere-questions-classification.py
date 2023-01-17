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
from sklearn.feature_extraction.text import TfidfVectorizer 

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation

from nltk.tokenize import word_tokenize

import re

import nltk

from nltk.corpus import stopwords

import string

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.stem.porter import *

from sklearn.model_selection import train_test_split





from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv(r'/kaggle/input/train.csv')

test = pd.read_csv(r'/kaggle/input/test.csv')
print(train.shape),print(test.shape), print(train['target'].value_counts())

train.head()
test.head()
all_questions =' '.join(train['question_text'])

question_1 =" ".join(train.loc[train['target']==1,'question_text'])

question_0 =" ".join(train.loc[train['target']==0,'question_text'])
wordcloud = WordCloud().generate(all_questions)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(question_1)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(question_0)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
stemmer = PorterStemmer()

lemma = WordNetLemmatizer()

my_stop=set(stopwords.words('english')+list(punctuation))
# # remove special characters, numbers, punctuations

# train['refined_text'] = train['question_text'].str.replace("[^a-zA-Z#]", " ")

# #tokenization

# tokenized = train['refined_text'].apply(lambda x: x.split())



# #stemming

# stemmed = tokenized.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming



# for i in range(len(stemmed)):

#     stemmed[i] = ' '.join(stemmed[i])



# train['refined_text'] = stemmed

def split_into_lemmas(message):

    list_of_tokens = []

    message = message.lower()

    meassge = re.sub('[^a-z ]+'," ",message)

    words = word_tokenize(message)

    for word in words:

        if word in my_stop:

            continue

        if word.isalpha():

            list_of_tokens.append(lemma.lemmatize(word))

    return list_of_tokens
train.head()
train,val =train_test_split(train,test_size=0.2,random_state=2, stratify = train['target'])

x_train = train['question_text']

y_train = train['target']

x_val = val['question_text']

y_val = val['target']



x_train.shape, y_train.shape, x_val.shape, y_val.shape
tfidf= TfidfVectorizer(analyzer=split_into_lemmas,min_df=20,max_df=500,stop_words=my_stop)
tfidf.fit(x_train)
tfidf_train = tfidf.transform(x_train)

tfidf_val = tfidf.transform(x_val)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, roc_auc_score
clf= MultinomialNB(class_prior=[0.70,0.30])

clf.fit(tfidf_train, y_train)
predictions=pd.DataFrame(list(zip(y_val,clf.predict(tfidf_val))),columns=['real','predicted'])

print(pd.crosstab(predictions['real'],predictions['predicted']))

print(accuracy_score(y_val,clf.predict(tfidf_val)))

roc_auc_score(y_val,clf.predict(tfidf_val))