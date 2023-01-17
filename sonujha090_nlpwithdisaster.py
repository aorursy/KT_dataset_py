import pandas as pd

import numpy as np

import numpy

import re

from nltk.corpus import stopwords
import os

os.listdir("../input/nlp-getting-started")
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
valid = train[:6250]

valid.shape
print('shape of training data = {} and shape of validation data is {}'.format(train.shape,valid.shape))
train.columns.values
train_text = train['text']

valid_text = valid['text']
def text_to_words(raw_text):

    letter_only = re.sub("[^a-zA-Z]"," ", raw_text)

    words = letter_only.lower().split()

    useful_words = [w for w in words if w not in stopwords.words('english')]

    return ' '.join(useful_words)
clean_train_text = []

for i in range(0,len(train)):

    clean_train_text.append(text_to_words(train_text[i]))

clean_train_text[:5]
clean_valid_text = []

for i in range(0,len(valid)):

    clean_valid_text.append(text_to_words(valid_text[i]))

clean_valid_text[:5]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer='word',max_features=5000)

train_data_features = cv.fit_transform(clean_train_text)

train_data_features = train_data_features.toarray()
valid_data_features = cv.transform(clean_valid_text)

valid_data_features = valid_data_features.toarray()
print(train_data_features.shape)

print(valid_data_features.shape)
dist = np.sum(train_data_features,axis=0)

for tag,count in zip(vocab,dist):

    print(count,tag)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(train_data_features,train['target'])
test_data_features.shape
from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(result,valid['target']))

print(accuracy_score(result,valid['target']))
test_text = test['text']

clean_test_review = []

for i in range(0,len(test)):

    clean_test_review.append(text_to_words(test_text[i]))

clean_test_review[:5]
# Get a bag of words for the test set, and convert to a numpy array

test_data_features = cv.transform(clean_test_review)

test_data_features = test_data_features.toarray()
result = classifier.predict(test_data_features)
submition_file = pd.DataFrame({'target':result},index=test['id'])
submition_file.to_csv('submission.csv')