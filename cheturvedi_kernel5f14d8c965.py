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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

corpus = data['text'].to_list()

target = data['target'].to_list()



test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test_corpus = test['text'].to_list()

text_id = test['id'].to_list()

len(corpus) == len(target)
text_clf = Pipeline([

    ('vect', CountVectorizer(stop_words='english')),

    ('tfidf', TfidfTransformer()),

    ('clf', MultinomialNB())

])

mlPipe = text_clf.fit(corpus,target)
y_predicted = mlPipe.predict(test_corpus)
len(y_predicted) == len(text_id)
#ans = pd.DataFrame (text_id,y_predicted)

#ans.head()

ans = pd.DataFrame(text_id,columns = ['id'])

ans['target'] = y_predicted

ans.to_csv('NaiveBayesWithoutStopWords.csv',index=False)