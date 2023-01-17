# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

data=pd.read_csv('../input/train.csv')

data.shape





# Any results you write to the current directory are saved as output.
data.head()
# Check for missing values

data.isnull().sum()
import nltk

import numpy as np

import pandas as pd
!pip install wordcloud
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

target1=data[data['target']==1]

target0=data[data['target']==0]
docs1=target1['question_text']

print(len(docs1))

docs0=target0['question_text']

print(len(docs0))
stopwords=nltk.corpus.stopwords.words('english')

wc1 = WordCloud(background_color='white',stopwords=stopwords).generate(' '.join(docs1))

plt.imshow(wc1)
wc0 = WordCloud(background_color='white',stopwords=stopwords).generate(' '.join(docs0))

plt.imshow(wc0)
docs=data['question_text']

len(data)
# We pass a raw document and this function returns a clean document.

stopwords=nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_sentence)

docs_clean.head()
print(data['target'].value_counts())

print(pd.isnull(data).sum()) #check for missing values
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(docs_clean,

                                               data['target'],

                                               test_size=0.2,

                                               random_state=100)
#Count Vectorization

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=50).fit(train_x)

train_x = vectorizer.transform(train_x)

test_x = vectorizer.transform(test_x)
from sklearn.naive_bayes import MultinomialNB , BernoulliNB

from sklearn.metrics import f1_score

model_mnb = MultinomialNB().fit(train_x , train_y)

test_pred_mnb = model_mnb.predict(test_x)

print(f1_score(test_y , test_pred_mnb))
model_bnb = BernoulliNB().fit(train_x , train_y)

test_pred_bnb = model_bnb.predict(test_x)

print(f1_score(test_y , test_pred_bnb))
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(max_depth = 10).fit(train_x, train_y)

test_pred = model_dt.predict(test_x)

print(f1_score(test_y,test_pred))
test_data = pd.read_csv('../input/test.csv')

test_data.head()
test_docs = test_data['question_text']

test_docs = test_docs.str.lower().str.replace('[^a-z ]' , '')
test_docs_clean = test_docs.apply(clean_sentence)
test_docs_clean = vectorizer.transform(test_docs_clean)
test_docs_pred_mnb = model_mnb.predict(test_docs_clean)
data_submit = pd.DataFrame({'qid' : test_data['qid'] , 'prediction' : test_docs_pred_mnb})

data_submit.to_csv('submission.csv' , index = False)