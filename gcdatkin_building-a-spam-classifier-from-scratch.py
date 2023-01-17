# Data

import numpy as np

import pandas as pd



# NLP

import re

from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize



# Preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



# Modeling

from sklearn.svm import SVC

from sklearn.metrics import f1_score
data = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
data
data.drop([data.columns[col] for col in [2, 3, 4]], axis=1, inplace=True)
encoder = LabelEncoder()



data['v1'] = encoder.fit_transform(data['v1'])

class_mappings = {index: label for index, label in enumerate(encoder.classes_)}
class_mappings
# Take an email string and convert it to a list of stemmed words

def processEmail(contents):

    ps = PorterStemmer()

    

    contents = contents.lower()

    contents = re.sub(r'<[^<>]+>', ' ', contents)

    contents = re.sub(r'[0-9]+', 'number', contents)

    contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', contents)

    contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', contents)

    contents = re.sub(r'[$]+', 'dollar', contents)

    

    words = word_tokenize(contents)

    

    for i in range(len(words)):

        words[i] = re.sub(r'[^a-zA-Z0-9]', '', words[i])

        words[i] = ps.stem(words[i])

        

    words = [word for word in words if len(word) >= 1]

    

    return words
# Take a list of emails and get a dictionary of the most common words

def getVocabulary(emails, vocab_length):

    vocabulary = dict()

    

    for i in range(len(emails)):

        emails[i] = processEmail(emails[i])

        for word in emails[i]:

            if word in vocabulary.keys():

                vocabulary[word] += 1

            else:

                vocabulary[word] = 1

                

    vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)

    vocabulary = list(map(lambda x: x[0], vocabulary[0:vocab_length]))

    vocabulary = {index: word for index, word in enumerate(vocabulary)}

    

    return vocabulary
# Get a dictionary key given a value

def getKey(dictionary, val):

    for key, value in dictionary.items():

        if value == val:

            return key
# Get the indices of vocab words used in a given email

def getIndices(email, vocabulary):

    word_indices = set()

    

    for word in email:

        if word in vocabulary.values():

            word_indices.add(getKey(vocabulary, word))

    

    return word_indices
def getFeatureVector(word_indices, vocab_length):

    feature_vec = np.zeros(vocab_length)

    

    for i in word_indices:

        feature_vec[i] = 1

        

    return feature_vec
vocab_length = 2000
vocabulary = getVocabulary(data['v2'].to_list(), vocab_length)



emails = data['v2'].to_list()

emails = list(map(lambda x: processEmail(x), emails))
X = list(map(lambda x: getFeatureVector(getIndices(x, vocabulary), vocab_length), emails))

X = pd.DataFrame(np.array(X).astype(np.int16))
y = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = SVC()



model.fit(X_train, y_train)
model.score(X_test, y_test)
y_pred = model.predict(X_test)
f1_score(y_test, y_pred)