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

from sklearn.multioutput import MultiOutputClassifier
# Read Data

train = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

train.head()

train_id = train['ID']

test_id = test['ID']



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
data_copy, text_cols = add_text_features(text_column_name = 'ABSTRACT', 

                                     data_file = data, max_features = 150000, min_df = 5, max_df = .5,

                                    ngram_range = (1, 3), lowercase = True, sparse = True)
# Split the data back to train and test

X_train = data_copy[:train.shape[0], :]

y_train = data[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']].iloc[:train.shape[0]]



X_test = data_copy[train.shape[0]:, :]

y_test = data[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']].iloc[train.shape[0]:]
X_train
# Train model - Logistic Regression is a good option for Text classification problems

#model = linear_model.LogisticRegressionCV(penalty = 'l2', Cs = 10, max_iter = 5000).fit(X_train, y_train)

#model = linear_model.RidgeClassifierCV().fit(X_train, y_train)

from sklearn import naive_bayes



#model = MultiOutputClassifier(estimator = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)).fit(X_train, y_train)

model = MultiOutputClassifier(estimator = linear_model.LogisticRegressionCV(Cs = 10, cv = 5, n_jobs = -1, max_iter = 5000)).fit(X_train, y_train)
def get_preds_multioutput(predictions):

    return np.array([[val[1] for val in inner] for inner in predictions]).T



def convert_probs_to_labels(predictions, threshold = .5, labels = None):

    final = []

    for prediction in predictions:

        temp = (prediction > threshold)*1

        final.append(temp)

        

    return final



def predict_1(predictions, threshold=.5):

    preds = get_preds_multioutput(predictions)

    preds = convert_probs_to_labels(preds, threshold = threshold, labels = None)

    return np.array(preds)



#predict_1(model.predict_proba(X_test))
sub = pd.DataFrame()

sub['ID'] = test_id



preds = predict_1(model.predict_proba(X_test))

sub[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']] = model.predict(X_test).astype(int)

sub.to_csv('sub.csv', index = None)