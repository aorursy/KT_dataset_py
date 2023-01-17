import numpy as np

import pandas as pd

import spacy

from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
nlp = English()
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv', index_col = 'id')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv', index_col = 'id')

submit = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv', index_col = 'id')

train.head()
train.target.value_counts()
import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



punctuations = string.punctuation

nlp = spacy.load('en')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



parser = English()



def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



def clean_text(text):

    return text.strip().lower()
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range= (1, 1))
from sklearn.model_selection import train_test_split

X = train.text

y = train.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()



pipe = Pipeline([

    ("cleaner" , predictors()),

    ("vectorizer" , bow_vector),

    ("predictor" , clf)])

pipe.fit(X_train,y_train)
from sklearn import metrics

predicted = pipe.predict(X_test)
print("accuracy :",  metrics.accuracy_score(predicted, y_test))
submit.head()
test_text = test.text

submit.target = pipe.predict(test_text)

submit.head()
submit.to_csv('submit.csv')