import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.info()
train['text'].describe()
test.head()
test.info()
train['text_len'] = train['text'].apply(len)
train.head()
sns.distplot(train['text_len'],kde=False,bins=50)
train['text_len'].describe()
train.hist(column='text_len',by='target',bins=50,figsize=(12,4))
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

import string

from nltk.corpus import stopwords
def text_process(news) :

    """

    1. remove punctuation

    2. remove stop words

    3. return list of clean text words

    """

    

    nopunc = [char for char in news if char not in string.punctuation]

    

    nopunc = "".join(nopunc)

    

    return [words for words in nopunc.split() if words.lower() not in stopwords.words('english')]
pipeline = Pipeline({

    ('bow',CountVectorizer(analyzer = text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

})
pipeline.fit(train['text'],train['target'])
pred = pipeline.predict(test['text'])
submission = pd.DataFrame(pred)

submission = pd.concat([test['id'],submission],axis=1)

submission.columns = ['id','target']
submission.to_csv('submission.csv',index=False)