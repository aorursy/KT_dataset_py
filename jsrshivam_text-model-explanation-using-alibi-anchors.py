!pip install alibi

!pip install lime
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import spacy

from alibi.explainers import AnchorText

from alibi.datasets import fetch_movie_sentiment

from alibi.utils.download import spacy_model

import pandas as pd

import sklearn

import sklearn.ensemble

import sklearn.metrics

from lime import lime_text

from sklearn.pipeline import make_pipeline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
movies = fetch_movie_sentiment()

movies.keys()
data = movies.data

labels = movies.target

target_names = movies.target_names
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)

train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)

train_labels = np.array(train_labels)

test_labels = np.array(test_labels)

val_labels = np.array(val_labels)
vectorizer = CountVectorizer(min_df=1)

vectorizer.fit(train)
np.random.seed(0)

clf = LogisticRegression(solver='liblinear')

clf.fit(vectorizer.transform(train), train_labels)
predict_fn = lambda x: clf.predict(vectorizer.transform(x))
preds_train = predict_fn(train)

preds_val = predict_fn(val)

preds_test = predict_fn(test)

print('Train accuracy', accuracy_score(train_labels, preds_train))

print('Validation accuracy', accuracy_score(val_labels, preds_val))

print('Test accuracy', accuracy_score(test_labels, preds_test))
model = 'en_core_web_md'

spacy_model(model=model)

nlp = spacy.load(model)
explainer = AnchorText(nlp, predict_fn)
class_names = movies.target_names
text = data[4]

print(text)

pred = class_names[predict_fn([text])[0]]

alternative =  class_names[1 - predict_fn([text])[0]]

print('Prediction: %s' % pred)
np.random.seed(0)

explanation = explainer.explain(text, threshold=0.95, use_unk=True)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))

print('Precision: %.2f' % explanation.precision)

print('\nExamples where anchor applies and model predicts %s:' % pred)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))

print('\nExamples where anchor applies and model predicts %s:' % alternative)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))

np.random.seed(0)

explanation = explainer.explain(text, threshold=0.95, use_unk=False, sample_proba=0.5)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))

print('Precision: %.2f' % explanation.precision)

print('\nExamples where anchor applies and model predicts %s:' % pred)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))

print('\nExamples where anchor applies and model predicts %s:' % alternative)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
explanation = explainer.explain(data[8], threshold=0.95, use_similarity_proba=True, use_unk=False,

                                sample_proba=0.5, top_n=20, temperature=0.2)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))

print('Precision: %.2f' % explanation.precision)

print('\nExamples where anchor applies and model predicts %s:' % pred)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))

print('\nExamples where anchor applies and model predicts %s:' % alternative)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
np.random.seed(0)

explanation = explainer.explain(text, threshold=0.95, use_similarity_proba=True, sample_proba=0.5,

                                use_unk=False, top_n=20, temperature=.2)



print('Anchor: %s' % (' AND '.join(explanation.anchor)))

print('Precision: %.2f' % explanation.precision)

print('\nExamples where anchor applies and model predicts %s:' % pred)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))

print('\nExamples where anchor applies and model predicts %s:' % alternative)

print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))