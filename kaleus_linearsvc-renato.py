!pip install pandarallel

!python -m spacy download pt
import pandas as pd

import numpy as np

import re

import string

import spacy

import pickle

import os

import matplotlib.pyplot as plt



from scipy.sparse import hstack

from nltk.corpus import stopwords



from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer,RSLPStemmer

from pandarallel import pandarallel



pandarallel.initialize(shm_size_mb=2000, progress_bar=True, nb_workers=3)

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 200)

%matplotlib inline
import nltk

nltk.download('stopwords')

nltk.download('rslp')

nltk.download('punkt')

nltk.download('wordnet')



stemmer=RSLPStemmer()

lemma=WordNetLemmatizer()
#Load data

train = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/train.csv')

test = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/test.csv')
# Count words occurrence on title column

pd.Series(' '.join(train['title']).lower().split()).value_counts()[:100]
# Search for word on title column

teste  = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/train.csv', encoding="utf-8").title.str.count('Proteçao')

teste.sum(axis = 0, skipna = True) 
# List rows filter by category

engine = pd.read_csv('/kaggle/input/nlp-deeplearningbrasil-ml-challenge/train.csv', encoding="utf-8")[train['category'] == 'CELLPHONE_REPAIR_TOOL_KITS']
engine[0:50]
# Words with incorrect spelling

#misspeling = {

    #"und": "unidade", 

    #"seg": "segundos", 

    #"lamp": "lâmpada",

    #"dobradica": "dobradiça", 

    #"profisional": "profissional", 

    #"recpetor": "receptor"

#}
# Custom stop words list

#remove_words = ['balada', 'segundos', 'profissional', 'golf']
#Category count

train['category'].value_counts()
#Add new features

#train['title_size'] = train['title'].progress_map(lambda s: len(s))

#train['title_upper'] = train['title'].progress_map(lambda s: sum(1 for c in s if c.isupper()))

#train['word_count'] = train['title'].progress_map(lambda s: len(s.split()))

#train['root_element'] = train['title'].progress_map(lambda s: ' '.join(token.lemma_ for token in nlp(s) if token.dep_ == 'ROOT'))
# Pre processing data



import unidecode

import unicodedata



nlp = spacy.load('pt', parser=False, entity=False)



def preProcessing(x):

    

    #has = False

    

    # Converting to Lowercase

    x = x.lower()

            

    #Remove stop words spacy

    x = ' '.join(token.text for token in nlp(x) if not token.is_stop)

    

    # Remove links

    x = re.sub(r'http\S+', ' ', x)

    

    # Remove all the special characters

    x = re.sub(r'\W', ' ', x)

    

    # Remove punct

    x = x.translate(str.maketrans("","", string.punctuation))

    

    # Remove Numbers

    x = re.sub(r'\d+', ' ', x)

    

    # Remove accents

    x = unicodedata.normalize('NFD', x)

    x = x.encode('ascii', 'ignore')

    x = x.decode("utf-8")

    

    # Stemmer

    result = []

    for item in x.split():

        result.append(stemmer.stem(item))

    

    x = ' '.join(result)    

    

    return x
train.head()
# Executing text pre processing on train dataframe

train['title'] = train['title'].parallel_map(preProcessing)
# Executing text pre processing on test dataframe

test['title'] = test['title'].parallel_map(preProcessing)
#After pre processing

train.head(100)
# List of imports



from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import preprocessing

from imblearn.over_sampling import SMOTE

from matplotlib import pyplot as plt

from sklearn.svm import SVC
# Define y value

y = train['category'].values
# TEST PARAMS



from imblearn.pipeline import Pipeline



X_train, X_test, y_train, y_test = train_test_split(train['title'], y, test_size=0.15, random_state=0)



text_clf_red = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),

                       ('tfidf', TfidfTransformer(smooth_idf=False)),

                       ('clf', LinearSVC())

                       ])



text_clf_red.fit(X_train, y_train)

y_test_pred = text_clf_red.predict(X_test)

accuracy_score(y_test, y_test_pred)
#pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),

#                       ('tfidf', TfidfTransformer()),

#                       ('clf', LinearSVC())

#                       ])



#parameters = {

#    'vect__max_df': (0.5, 0.75, 0.9),

#    'vect__min_df':(2 , 3),

#    'vect__max_features': (25000, 35000, 40000),

    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams

#    'tfidf__use_idf': (True, False),

#    'tfidf__smooth_idf': (True, False),

#    'tfidf__sublinear_tf': (True, False),

    #'tfidf__norm': ('l1', 'l2'),

#    'clf__class_weight': (None, 'balanced'),

#    'clf__C': (1, 100, 500),

    #'clf__max_iter': (10, 50, 80, 150),

#}
#grid_search = GridSearchCV(pipeline, parameters, cv=6,

#                           n_jobs=-1, verbose=1)



#grid_search.fit(X_train, y_train)
#grid_search.best_params_
# Vectorizer

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html



# Value of max_feature is not the same, decrease to run on kaggle, original was 36500

vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))

con = pd.concat([train, test], sort=False)

X = vectorizer.fit(con['title'])



# Transform with train and test title

X_comp = vectorizer.transform(con['title'])

X = vectorizer.transform(train['title'])
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

tfidfconverter = TfidfTransformer(smooth_idf=False)

X_comp = tfidfconverter.fit(X_comp)

X = tfidfconverter.transform(X).toarray()
# Train test split on data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
#OVERSAMPLE



smote = SMOTE(ratio='minority')

X_res, y_res = smote.fit_sample(X, y)
%%time

# LINEAR SVC

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

linf = LinearSVC()

linf.fit(X_res, y_res)



# predict and evaluate predictions

predictions = linf.predict(X_test)



#linf_score = cross_val_score(linf, X_res, y_res, cv=5, scoring='accuracy')

#print(linf_score)

#print(linf_score.mean())



print('accuracy %s' % accuracy_score(predictions, y_test))

print(classification_report(y_test, predictions,target_names=train['category'].unique()))
simple_data = vectorizer.transform(test['title'])

simple_data = tfidfconverter.transform(simple_data).toarray()



simple_result = linf.predict(simple_data)

#y_test_pred = text_clf_red.predict(test['title'])



# Submission

submission = pd.DataFrame({

    "id": test.index.values,

    "category": simple_result

})



# Create file

submission.to_csv('submission.csv', index=False)
submission.head(10)