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
from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submit = pd.read_csv("../input/sample_submission.csv")
train.shape
test.shape
submit.shape
train.isnull().sum()
train.Word[train["Word"] == '.']
train.pop("id")

train.head()
from nltk.probability import FreqDist

fdist = FreqDist(train["Word"])

fdist

'''import nltk

nltk.pos_tag(train["Word"][0])'''
train.dropna(inplace=True)
'''from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.transform(train["Word"])'''
train.tag.unique()
train.tag[train.tag == 'O']=1

train.tag[train.tag == 'B-indications']=2

train.tag[train.tag == 'I-indications']=3
y = train["tag"]

y=y.astype('int')
test.fillna("None", inplace=True)
test.shape
test.head()
'''cv = CountVectorizer()

text_counts= cv.fit_transform(train["Word"])

text_counts'''
le = LabelEncoder()

train["Word"] = le.fit_transform(train["Word"])



test["Word"] = le.fit_transform(test["Word"])
x = train.drop(labels = "tag", axis = 1)
test.pop("id")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.metrics import f1_score

clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)

print("MultinomialNB Accuracy:",metrics.f1_score(y_test, predicted, average = 'weighted'))
'''cv = CountVectorizer()

text_counts1= cv.fit_transform(test["Word"])

'''
pred= clf.predict(test)
submit.shape
len(pred)

2994463

2994370
submit["tag"] = pred 
submit.head()
submit.tag[submit.tag == 1]='O'
submit.head()
submit.to_csv("submit.csv", index=False)