from itertools import *

import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords 

from scipy import sparse

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import re, string

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Activation, Embedding

from keras.preprocessing import text, sequence
#Loading the train and test files, as usual



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#A sneak peek at the training and testing dataset

train.head()



#preprocessing step is to check for nulls



train.isnull().any(),test.isnull().any()

types = ['toxic', 'severe_toxic', 'obscene', 'threat',

       'insult', 'identity_hate']

count_list = []

for i in types:

    count_list.append(train[i].sum())

    

sum=sum(count_list)

sum
import string

import itertools 

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation

from nltk.stem import PorterStemmer





label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

label_train = train[label_list]



def clean_text(text):

    txt = str(text)

    # Remove all symbols

    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)

    txt = re.sub(r'\n',r' ',txt)

    # Replace words like sooooooo with so

    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

    # Remove urls and emails

    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)

    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)

	# Remove punctuation from text

    txt = ''.join([c for c in text if c not in punctuation])

    #Convert to lowercase

    txt = " ".join([w.lower() for w in txt.split()])

    #Remove stopwords   

    stop_words = set(stopwords.words('english'))

    txt = " ".join([w for w in txt.split() if w not in stop_words])

    return txt



def sequential_model():

#preprocessing for removing stopwords

	train['comment_text'] = train['comment_text'].map(lambda x: clean_text(x))

	label_train = train[label_list].values

	test['comment_text']=test['comment_text'].map(lambda x: clean_text(x))

	max_features = 20000

	max_text_length = 300

	filters = 250

	kernel_size = 3



	#tokenizer

	x_tokenizer = text.Tokenizer(num_words=max_features)

	x_tokenizer.fit_on_texts(train['comment_text'])

	x_tokenized = x_tokenizer.texts_to_sequences(train['comment_text'])

	x_test_tokenized = x_tokenizer.texts_to_sequences(test['comment_text'])

	x_train_val = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)

	x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)



	#train - validation split

	x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, label_train, test_size=0.1, random_state=1)

	#defining sequential model

    #defining sequential model

	classifier_model = Sequential()

	classifier_model.add(Embedding(max_features, 50, input_length=max_text_length))

	#classifier_model.add(Dropout(0.2))

	classifier_model.add(Conv1D(filters, kernel_size, activation='relu',  padding='valid', strides=1))

	classifier_model.add(GlobalMaxPooling1D(5))

	classifier_model.add(Dense(100))

	classifier_model.add(Dropout(0.2))

	classifier_model.add(Activation('sigmoid'))

	classifier_model.add(Dense(6))

	classifier_model.add(Activation('softmax'))





	#compilation and metrics definition

	classifier_model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')

	#classifier_model.summary()

	classifier_model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1,

	                     validation_data=(x_validation, y_validation))





	#predict model on testing data set

	y_prediction = classifier_model.predict(x_testing, verbose=1)

    #read sample submission csv and write predicted values to a new csv

	sample_submission = pd.read_csv("../input/sample_submission.csv")

	sample_submission[label_list] = y_prediction

	sample_submission.to_csv("results.csv", index=False)



if __name__ == "__main__":

	#ensemble_methods()

	sequential_model()