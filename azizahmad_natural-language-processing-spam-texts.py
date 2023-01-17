#import NLTK library and pandas

import nltk

import pandas as pd
# read the csv file to a dataframe

messages = pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
# first 5 rows of the dataframe

messages.head(5)
# to clean the text from commonly used words and punctuation, we import these

import string

from nltk.corpus import stopwords
# create a function that can take in a message and clean it, removing puncutation and "stopwords", then returning the "cleaned" version

def text_process(mess):

    """

    1. remove punctuation

    2. remove stop words

    3. return list of clean text

    """

    

    nopunc = [letter for letter in mess if letter not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
# CountVectorizer coverts the messages into 'bags of words', which build a large sparse matrix for the entire message set

# Term Frequency, Inverse Document Frequency measures the importance of each word in the message and compare it against all the messages to detemine its importance

# we use Naive Bayes for the text classifier

# Pipeline is used to simplify the action of feeding in training and test data for cross validation

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
# A pipleine estimator is built, combining the above processes

pipeline = Pipeline([

    ('bag of words',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifer',MultinomialNB())

])
# Split our data into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(messages['Message'],messages['Category'],test_size=0.33)
# Fit the data to the pipeline estimator

pipeline.fit(X_train,y_train)
# We create predictions

pred = pipeline.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
# Here's the results of how well our model performed at predicitng 'spam' messages from 'ham' (regular) messages

print(confusion_matrix(y_test,pred))

print('')

print(classification_report(y_test,pred))