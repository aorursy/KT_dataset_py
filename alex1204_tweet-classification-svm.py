import numpy as np # linear algebra

import pandas as pd

import nltk

import spacy

import re

import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer

from stop_words import get_stop_words
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
train_df
docs = train_df.text
train_target = train_df.target
nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)
normalized_docs = []

for doc in docs:

    doc = doc.lower() #verkleinen

    doc = re.sub(r'[0-9]', '', doc) # verwijderen cijfers

    doc = re.sub(r'[&(),.#://?!]', '', doc) # verwijderen speciale tekens

    doc = re.sub('-', ' ', doc) # verwijderen van het verlengteken

    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc) # Verwijderen extra newlines

    doc = re.sub(' +', ' ', doc) # Verwijderen overbodige whitespace

    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore') # verwijderen accented tekens

    doc = nlp(doc)

    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]) # Lemmatizeren van woorden

    #ps = nltk.porter.PorterStemmer()

    #doc = ' '.join([ps.stem(word) for word in doc.split()])

    normalized_docs.append(doc)

print('Docs normalized')
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
new_docs = test_df.text
normalized_new_docs = []

for doc in new_docs:

    doc = doc.lower() #verkleinen

    doc = re.sub(r'[0-9]', '', doc) # verwijderen cijfers

    doc = re.sub(r'[&(),.]#', '', doc) # verwijderen speciale tekens

    doc = re.sub('-', ' ', doc) # verwijderen van het verlengteken

    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc) # Verwijderen extra newlines

    doc = re.sub(' +', ' ', doc) # Verwijderen overbodige whitespace

    doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore') # verwijderen accented tekens

    doc = nlp(doc)

    doc = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]) # Lemmatizeren van woorden

    #ps = nltk.porter.PorterStemmer()

    #doc = ' '.join([ps.stem(word) for word in doc.split()])

    normalized_new_docs.append(doc)

print('New docs normalized')
import warnings

warnings.filterwarnings('ignore')
from pprint import pprint

from time import time

import logging



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', LinearSVC()),

])
parameters = {

    'vect__max_df': (0.5, 0.75, 1.0),

    'vect__max_features': (None, 1000, 10000),

    'vect__ngram_range': ((1, 1), (1, 2), (1,3)),  # unigrams or bigrams or trigrams

    'tfidf__use_idf': (True, False),

    'tfidf__norm': ('l1', 'l2'),

    'clf__C': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-4)

}
if __name__ == "__main__":

    # multiprocessing requires the fork to happen in a __main__ protected

    # block



    # find the best parameters for both the feature extraction and the

    # classifier

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)



    print("Performing grid search...")

    print("pipeline:", [name for name, _ in pipeline.steps])

    print("parameters:")

    pprint(parameters)

    t0 = time()

    grid_search.fit(normalized_docs, train_target)

    print("done in %0.3fs" % (time() - t0))

    print()



    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))
predictions = grid_search.predict(normalized_new_docs)
output = pd.DataFrame({'id':test_df.id, 'target':predictions})

output
output.to_csv('submission.csv', index=False)