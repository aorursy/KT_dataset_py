# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load ML Pkgs

# ML Packages

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.metrics import accuracy_score 

from sklearn.base import TransformerMixin 

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data=pd.read_csv('../input/nlp-getting-started/train.csv')

data.head()

data.shape
# Load NLP pkgs

import spacy

import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

nlp = spacy.load('en_core_web_sm')

# Use the punctuations of string module

punctuations = string.punctuation

parser = English()

stopwords = list(STOP_WORDS)
#Custom transformer using spaCy 

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):

        return self

    def get_params(self, deep=True):

        return {}
# Basic function to clean the text 

def clean_text(text):  

    return text.strip().lower()
def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    return mytokens
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 

#classifier = LinearSVC()

classifier = SVC(C=10, gamma=2e-2, probability=True)
# Using Tfidf

tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)
# Splitting Data Set

from sklearn.model_selection import train_test_split

# Features and Labels

X = data['text']

ylabels = data['target']



X_train, X_val, y_train, y_val = train_test_split(X, ylabels, test_size=0.2, random_state=42)
# Create the  pipeline to clean, tokenize, vectorize, and classify 

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', vectorizer),

                 ('classifier', classifier)])
# Fit our data

pipe.fit(X_train,y_train)
print("Accuracy Score:",pipe.score(X_val, y_val))
submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission
# Read the test data



test=pd.read_csv('../input/nlp-getting-started/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.



test_X = test['text']

# Use the model to make predictions

predicted = pipe.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted)



my_submission = pd.DataFrame({'id': test.id, 'target': predicted})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)