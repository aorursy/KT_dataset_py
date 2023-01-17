# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.tsv.zip',sep='\t')
data.info()
data.head()
print(data.iloc[0]['Phrase'],'Sentiment - ',data.iloc[0]['Sentiment'])
print(data.iloc[1]['Phrase'],'Sentiment - ',data.iloc[1]['Sentiment'])
print(data.iloc[32]['Phrase'],'Sentiment - ',data.iloc[32]['Sentiment'])

print('\n')

print(data.iloc[33]['Phrase'],'Sentiment - ',data.iloc[33]['Sentiment'])
import string

string.punctuation
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()



def own_analyser(phrase):

    phrase = phrase.split()

    for i in range(0,len(phrase)):

        k = phrase.pop(0)

        if k not in string.punctuation:

                phrase.append(lm.lemmatize(k).lower())    

    return phrase
data.columns
X = data['Phrase']

y = data['Sentiment']
from sklearn.model_selection import train_test_split

phrase_train,phrase_test,sentiment_train,sentiment_test = train_test_split(X,y,test_size=0.3)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline



pipeline = Pipeline([('BOW',CountVectorizer(analyzer=own_analyser)),

                    ('tfidf',TfidfTransformer()),

                    ('classifier',MultinomialNB())])
pipeline.fit(phrase_train,sentiment_train)
predictions = pipeline.predict(phrase_test)
from sklearn.metrics import classification_report
print(classification_report(sentiment_test,predictions))
test_data = pd.read_csv('../input/test.tsv.zip',sep='\t')
test_data.head()
test_predictions = pipeline.predict(test_data['Phrase'])
phrase_id = test_data['PhraseId'].values
test_predictions.shape
final_answer = pd.DataFrame({'PhraseId':phrase_id,'Sentiment':test_predictions})
final_answer.head()
filename = 'Sentiment Analysis - NaiveBayes.csv'



final_answer.to_csv(filename,index=False)



print('Saved file: ' + filename)