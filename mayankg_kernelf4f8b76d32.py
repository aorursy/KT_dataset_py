import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup #Used to deal with HTML and XML tags

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re



import sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("../input/labeledTrainData.tsv", sep = "\t")

validation = pd.read_csv("../input/testData.tsv", sep = "\t")

sampleSubmission = pd.read_csv("../input/sampleSubmission.csv")
stopwords = set(stopwords.words('english'))

stopwords.remove('no')

stopwords.remove('not')



lemma = WordNetLemmatizer
def clean_text(text):

    remove_html_tags = BeautifulSoup(text).get_text()

    remove_punctuation = re.sub("[^a-zA-Z\s?!]", " ",remove_html_tags)

    lowercase = word_tokenize(remove_punctuation.lower())

    removed_stopwords = [w for w in lowercase if w not in stopwords]

    lemmatized = [lemma.lemmatize('v',word) for word in removed_stopwords]

    cleaned_review = " ".join(lemmatized)

    return cleaned_review
train['clean_text'] = train['review'].apply(lambda x: clean_text(x))

train['text_length'] = train['clean_text'].apply(lambda x: len(x))

train['?marks'] = train['clean_text'].apply(lambda x: len(re.findall("(\?)",x)))

train['!marks'] = train['clean_text'].apply(lambda x: len(re.findall("(\!)",x)))
vectorizer = CountVectorizer(analyzer = "word",

                             tokenizer = None,

                             preprocessor = None,

                             stop_words = None,

                             max_features = 5000)



train_data_features = vectorizer.fit_transform(train.clean_text)
train_data_features = train_data_features.toarray()

train_df = pd.DataFrame(train_data_features)

train_df = pd.concat([train_df,train[['text_length','?marks','!marks']]],axis=1)
model = RandomForestClassifier(n_estimators = 100) 

model = model.fit(train_df, train["sentiment"] )
validation['clean_text'] = validation['review'].apply(lambda x: clean_text(x))

validation['text_length'] = validation['clean_text'].apply(lambda x: len(x))

validation['?marks'] = validation['clean_text'].apply(lambda x: len(re.findall("(\?)",x)))

validation['!marks'] = validation['clean_text'].apply(lambda x: len(re.findall("(\!)",x)))
validation_data_features = vectorizer.transform(validation.clean_text)

validation_data_features = validation_data_features.toarray()

test_df = pd.DataFrame(validation_data_features)

test_df = pd.concat([test_df,validation[['text_length','?marks','!marks']]],axis=1)
result_test = model.predict(test_df)

output = pd.DataFrame( data={"id":validation["id"], "sentiment":result_test} )

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
