from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
import numpy as np

import pandas as pd
import string
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
def cleaner(x):

    return [a for a in (''.join([a for a in x if a not in string.punctuation])).split() if a.lower() not in stopwords.words('english')]
lr = LogisticRegression(solver='liblinear', random_state=777) 
Pipe = Pipeline([

    ('bow',CountVectorizer(analyzer=cleaner)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB())

])
train.drop('id',inplace = True,axis = 1)

test1 = test.drop('id',axis = 1)
train.fillna('None',inplace=True)

test1.fillna('None',inplace=True)

test.fillna('None',inplace=True)
train
Pipe.fit(train[['text','keyword','location']].values,train['target'])
pred2 = Pipe.predict(test[['text','keyword','location']].values)



submit_pred = pd.Series(pred2)
submit_pred == pred2
Submission = pd.DataFrame({"id" : test["id"], "target":pred2})

Submission 



Submission.to_csv('submission.csv',index=False)