import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn import preprocessing, linear_model, ensemble, metrics, model_selection, svm, pipeline, naive_bayes

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk

import spacy

import textblob

from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer
path = '\\kaggle\\input\\janata-hacknlp\\'
# Read Data

train = pd.read_csv('/kaggle/input/janata-hacknlp/train_E52nqFa/train.csv')

test = pd.read_csv('/kaggle/input/janata-hacknlp/test_BppAoe0/test.csv')

meta = pd.read_csv('/kaggle/input/janata-hacknlp/train_E52nqFa/game_overview.csv')



# Rename Certain columns

train = train.rename({'year': 'year_no', 'title': 'title_no'}, axis = 1)

test = test.rename({'year': 'year_no', 'title': 'title_no'}, axis = 1)

meta = meta.rename({'title': 'title_no', 'developer': 'developer_no'}, axis = 1)
train.head()
meta.head()
# Add Meta Data

train = train.merge(meta, on = 'title_no')

test = test.merge(meta, on = 'title_no')



train_id = train.review_id

test_id = test.review_id



# Create indices to split train and test on later

train['train_ind'] = np.arange(train.shape[0])

test['train_ind'] = np.arange(train.shape[0], train.shape[0]+test.shape[0])



# Merge Train and Test - This approach only works for competitions - not for model deployment in real projects.

data = pd.concat([train, test], axis = 0)
# Create class which performs Label Encoding - if required

class categorical_encoder:

    def __init__(self, columns, kind = 'label', fill = True):

        self.kind = kind

        self.columns = columns

        self.fill = fill

        

    def fit(self, X):

        self.dict = {}

        self.fill_value = {}

        

        for col in self.columns:

            label = preprocessing.LabelEncoder().fit(X[col])

            self.dict[col] = label

            

            # To fill

            if self.fill:

                self.fill_value[col] = X[col].mode()[0]

                X[col] = X[col].fillna(self.fill_value[col])

                

        print('Label Encoding Done for {} columns'.format(len(self.columns)))

        return self

    def transform(self, X):

        for col in self.columns:

            if self.fill:

                X[col] = X[col].fillna(self.fill_value[col])

                

            X.loc[:, col] = self.dict[col].transform(X[col])

        print('Transformation Done')

        return X
# Create Lemmatizer - if required

class LemmaTokenizer(object):

    def __init__(self):

        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):

        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
# Function to Create CountEncoded and tf-idf features

def add_text_features(text_column_name, data_file, max_features = 2000, txn = 'tf-idf', min_df = 1, max_df = 1.0,

                     ngram_range = (1, 1), lowercase = True, sparse = False, tokenizer = None):

    if txn == 'count':

        # Use Count Vectorizer

        counts = CountVectorizer(max_features = max_features, min_df = min_df, 

        max_df = max_df, ngram_range = ngram_range, lowercase = lowercase, tokenizer=tokenizer).fit(data_file[text_column_name])

    if txn == 'tf-idf':

        counts = pipeline.make_pipeline(CountVectorizer(max_features = max_features, min_df = min_df, 

        max_df = max_df, ngram_range = ngram_range, lowercase = lowercase, tokenizer=tokenizer),

                                        TfidfTransformer()).fit(data_file[text_column_name])

    text_features = counts.transform(data_file[text_column_name])

    

    # Return for sparse output

    if sparse: return text_features, None

    

    # Create Mapping

    if txn == 'count':

        mapping = {val: key for key, val in counts.vocabulary_.items()}

    if txn == 'tf-idf':

        mapping = {val: key for key, val in counts['countvectorizer'].vocabulary_.items()}

    

    # Create DataFrame

    text_features_data = pd.DataFrame(text_features.toarray())

    text_features_data = text_features_data.rename(mapping, axis = 1)

    text_cols = text_features_data.columns.tolist()

    

    # Append to dataframe

    data_copy = pd.concat([data_file.reset_index(drop = True), text_features_data.reset_index(drop = True)], axis = 1)

    return data_copy, text_cols
# Label Encode Certain columns - for use later

enc = categorical_encoder(columns = ['title_no','developer_no', 'publisher']).fit(data)

data_copy = enc.transform(data)
data_copy, text_cols = add_text_features(text_column_name = 'user_review', 

                                     data_file = data_copy, max_features = 120000, min_df = 5, max_df = .5,

                                    ngram_range = (1, 4), lowercase = True, sparse = True, tokenizer = LemmaTokenizer())
# Split the data back to train and test

X_train = data_copy[:train.shape[0], :]

y_train = data['user_suggestion'].iloc[:train.shape[0]]



X_test = data_copy[train.shape[0]:, :]

y_test = data['user_suggestion'].iloc[train.shape[0]:]
print(X_train.shape, X_test.shape)
# Train model - Logistic Regression is a good option for Text classification problems

model = linear_model.LogisticRegressionCV(penalty = 'l2', Cs = 10, max_iter = 5000).fit(X_train, y_train)

sub = pd.DataFrame()

sub['review_id'] = test_id

#sub['user_suggestion'] = (model.predict_proba(X_test)[:, 1]>.50).astype(int)

sub['user_suggestion'] = model.predict(X_test).astype(int)

sub['user_suggestion'].value_counts()

sub.to_csv('sub.csv', index = None)