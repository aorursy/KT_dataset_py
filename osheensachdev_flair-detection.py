import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
def get_data(train_file, test_file = None):

    if test_file == None:

        frame = pd.read_csv(train_file)

        print(frame.head(5))

        data = frame.values

        np.random.shuffle(data)

        return data

    else:

        train_frame = pd.read_csv(train_file)

        test_frame = pd.read_csv(test_file, error_bad_lines=False, quoting = 2)



        train_data = train_frame.values

        test_data = test_frame.values

        np.random.shuffle(train_data)

        np.random.shuffle(test_data)



        return train_data, test_data



def get_training_testing_sets(train_file, test_file = None):

    if test_file == None:

        data = get_data(train_file)

        train_data, test_data = train_test_split(data)

    else:



        train_data, test_data = get_data(train_file, test_file)



    X_train = train_data[:, 1:]

    Y_train = train_data[:, :1]

    X_test = test_data[:, 1:]

    Y_test = test_data[:, :1]



    print(X_train.shape, X_test.shape)

    

    return X_train, Y_train, X_test, Y_test



data = get_data('../input/Subreddit_India.csv')
import re

import string



import nltk

from nltk.tokenize.casual import _replace_html_entities

# refer: http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer

# from utils.contractions import expand_contractions

import spacy

nlp = spacy.load('en_core_web_sm')

stopwords = spacy.lang.en.stop_words.STOP_WORDS
def cleanup(text):

    text = text.lower()

    text = _replace_html_entities(text) # fix HTML character entities

#     text = expand_contractions(text) # expand contractions

    text = re.sub(r'@\S+', '', text) # remove mentions

    text = re.sub(r'(www\.\S+)|(https?\://\S+)', '', text) # remove urls

    text = re.sub(r'#(\S+)', r'\1', text) # replaces #hashtag with hashtag

    text = re.sub(r'\brt\b', '', text) # remove RT

    text = re.sub(r'\'s', '', text) # remove possession apostrophe

    text = re.sub(r"n\'t", " not", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'t", " not", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'m", " am", text)

    text = re.sub(r'(\.{2,})|(\s+)', ' ', text) # replace 2+ dots/spaces with a single space

    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non-alphanumeric chars (punctuations also removed)

    text = re.sub(r'(.)\1+', r'\1\1', text) # replace repeated char seq of length >=2 with seq of length 2

    return text



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

import string

from nltk.stem import WordNetLemmatizer





punctuations = string.punctuation

lemmatizer = WordNetLemmatizer()



def preprocess(sentence):

    if str(sentence) == 'nan':

        return ''

    sentence = cleanup(sentence)

    sentence = ''.join(j for j in sentence if j not in punctuations)

    sentence = ' '.join(lemmatizer.lemmatize(j.lower()) for j in sentence.split())

    return sentence
print(data[0])
for row in range(len(data)):

    data[row][3] = preprocess(data[row][3]) + ' ' + preprocess(data[row][4])

    data[row][4] = ''
print(data[0])
from sklearn.preprocessing import LabelEncoder



X = data[:, 3]



encoder = LabelEncoder()

Y = encoder.fit_transform(data[:, 1:2])

Y = np.array(Y)

print(Y[0])





X_test = X[:1500]

X_train = X[1500:]



Y_test = Y[:1500]

Y_train = Y[1500:]



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline



from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

import warnings



print(encoder.classes_)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    for clf, name in [(OneVsRestClassifier(BernoulliNB()),'BernoulliNB'), (RandomForestClassifier(max_depth = 30), 'RandomForestClassifier', ), (OneVsRestClassifier(LogisticRegression(C = 0.2)), 'LogisticRegression'), (OneVsRestClassifier(LinearSVC(C = 0.05)), 'SVC') ]:

#     for clf, name in [(OneVsRestClassifier(LinearSVC(C = 0.06)), 'SVC') ]:

        print(name)

    #     Y_train.reshape(Y_train.shape[0],)

    #     Y_test.reshape(Y_test.shape[0])

        clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', clf)])

        clf.fit(X_train, Y_train)

        

        predictions = clf.predict(X_train)

        accuracy = accuracy_score(Y_train, predictions)

        print('training accuracy :',accuracy)



        predictions = clf.predict(X_test)

#         print(predictions[1], Y_test[1])

        accuracy = accuracy_score(Y_test, predictions)

        print('testing accuracy :',accuracy)



        print(classification_report(Y_test, predictions))
import pickle

pickle.dump(clf, open('model', 'wb'))