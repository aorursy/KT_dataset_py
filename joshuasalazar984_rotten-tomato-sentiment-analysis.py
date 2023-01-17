import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
Train= pd.read_csv("../input/train.tsv", sep="\t")

Test = pd.read_csv("../input/test.tsv", sep="\t")

test_sentiment=pd.read_csv("../input/sampleSubmission.csv")
import string

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
Train.head()
Train['Text_length']=Train['Phrase'].apply(len)
Train['Text_length'].plot.hist(bins=50)
Train.hist(column='Text_length', by='Sentiment', bins=60, figsize=(12,4))
sns.countplot(Train['Sentiment'])
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, classification_report
def text_process(phrase):

    """

    1. remove punctuation

    2. remove stop words

    3. return list of stopwords

    """

    

    nopunc=[char for char in phrase if char not in string.punctuation]

    

    nopunc=''.join(nopunc)

    

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#Creating a pipeline using CountVectorizer(), TfidfTransformer(),MultinomialNB()



pipeline=Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),

    ('tfidf', TfidfTransformer()),

    ('Classifier', MultinomialNB())

])
#Train test split



X_train=Train['Phrase']

X_test=Test['Phrase']

y_train=Train['Sentiment']
pipeline.fit(X_train, y_train)
pipe_pred=pipeline.predict(X_test)
#Exporting our Submission



test_sentiment['Sentiment']=pipe_pred

test_sentiment.to_csv('test_sentiment.csv', sep=',', index=False)