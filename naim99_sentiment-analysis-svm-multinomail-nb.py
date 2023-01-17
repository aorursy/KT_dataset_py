#import libraries

import io 

import pandas as pd

trainG = pd.read_excel('../input/train-test-sentiment-analysis/trainG_clean.xlsx')

test = pd.read_excel('../input/train-test-sentiment-analysis/test_clean.xlsx') 
trainG
test = test[test['ItemID'].map(len) < 10]

test
data = trainG.copy()

data.drop('ItemID', axis=1, inplace=True)

print(data.isnull().any())

data['Sentiment'] = data['Sentiment'].astype('category')

print(type(data['Sentiment'][0]))

data['label_id'] = data['Sentiment'].cat.codes

data['label_id'].head()
import re



REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")



def preprocess_reviews(reviews):

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]

    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    

    return reviews



data['SentimentText']  = data['SentimentText'].astype(str)

test['SentimentText'] = test['SentimentText'].astype(str)

data['SentimentText'] = preprocess_reviews(data['SentimentText'])



test['SentimentText'] = preprocess_reviews(test['SentimentText'])
from sklearn import svm

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

from sklearn.metrics import classification_report,accuracy_score

import six.moves.cPickle as pickle



import numpy as np 

import pandas as pd



import spacy

from spacy.matcher import Matcher

from spacy.tokens import Span

from spacy import displacy
nlp=spacy.load("en_core_web_sm")
from spacy.lang.en.stop_words import STOP_WORDS

stopwords = list(STOP_WORDS)

import string

punct=string.punctuation



def text_data_cleaning(sentence):

    doc = nlp(sentence)

    

    tokens = []

    for token in doc:

        if token.lemma_ != "-PRON-":

            temp = token.lemma_.lower().strip()

        else:

            temp = token.lower_

        tokens.append(temp)

    

    cleaned_tokens = []

    for token in tokens:

        if token not in stopwords and token not in punct:

            cleaned_tokens.append(token)

    return cleaned_tokens
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
tfidf = TfidfVectorizer(tokenizer = text_data_cleaning)

classifier = LinearSVC()
x = trainG['SentimentText']

y = trainG['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
y_pred=clf.predict(test['SentimentText'])
y_pred
y_pred[1]
sub_file=pd.DataFrame({'id':test['ItemID'],'target':y_pred.round().astype(int)})
prediction = list() 

for i in range(len(test['SentimentText'])):

  prediction.append(y_pred[i]) 





test['prediction'] = prediction  




test.to_excel('resultat_SVM.xlsx' , index=False)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from multiprocessing import Pool
tfidf = TfidfVectorizer(encoding='utf-8',

                       ngram_range=(1,3),

                       max_df=1.0,

                       min_df=10,

                       max_features=500,

                       norm='l2',

                       sublinear_tf=True)
train_features = tfidf.fit_transform(X_train).toarray()

print(train_features.shape)
test_features = tfidf.transform(X_test).toarray()

print(test_features.shape)
train_labels = y_train

test_labels = y_test
import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
mnb_classifier = MultinomialNB()
mnb_classifier.fit(train_features, train_labels)
mnb_prediction = mnb_classifier.predict(test_features)
training_accuracy = accuracy_score(train_labels, mnb_classifier.predict(train_features))

print(training_accuracy)
testing_accuracy = accuracy_score(test_labels, mnb_prediction)

print(testing_accuracy)
print(classification_report(test_labels, mnb_prediction))
conf_matrix = confusion_matrix(test_labels, mnb_prediction)

print(conf_matrix)
test_vectorizer =tfidf.transform(test['SentimentText']).toarray()
test_vectorizer.shape
final_predictions = mnb_classifier.predict(test_vectorizer)
submission_df = pd.DataFrame()
submission_df['Id'] = test['ItemID']

submission_df['target'] = final_predictions
submission_df['target'].value_counts()
submission = submission_df.to_csv('resultat_NB.csv',index = False)
