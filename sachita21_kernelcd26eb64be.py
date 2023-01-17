# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.probability import FreqDist



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression



from sklearn import metrics



from nltk import pos_tag

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score



import string

import re



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Set random seed

#This is used to reproduce the same result every time if the script is kept consistent otherwise each run will produce different results. The seed can be set to any number.

np.random.seed(500)
# Get train and test data from csv file

train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
# Check few rows for train data set

train_df.head()
# Check column info for train data set

train_df.info()
# Check number of null values in each column for train data set

train_df.isnull().sum()
# Check few rows for test data set

test_df.head()
# Check column info for test data set

test_df.info()
# Check sum of null values in each column

test_df.isnull().sum()
# Check percentage of data for each target variable 

print(round(train_df.target.value_counts(normalize=True)*100, 2))
# data distribution for each target variable

sns.countplot(train_df.target)
# get column names for train data set

train_df.columns
# Creating a copy to clean text and keeping original

train_df['text_cleaned'] = train_df.text.copy()
# Functions to clean text

def clean_text(entry):

    #WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. 

    #By default it is set to Noun

    tag_map = defaultdict(lambda : wn.NOUN)

    tag_map['J'] = wn.ADJ

    tag_map['V'] = wn.VERB

    tag_map['R'] = wn.ADV

    Final_words = []

    # Initializing WordNetLemmatizer()

    word_Lemmatized = WordNetLemmatizer()

    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only alphabets

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words.append(word_Final)

    return str(Final_words)
# Step - a : Remove blank rows if any.

train_df['text_cleaned'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently

train_df['text_cleaned'] = [entry.lower() for entry in train_df['text_cleaned']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words

train_df['text_cleaned']= [word_tokenize(entry) for entry in train_df['text_cleaned']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

train_df['text_cleaned'] = train_df['text_cleaned'].apply(lambda x: clean_text(x))
# Check train data set after cleaning

train_df.head()
# Step - a : Remove blank rows if any.

test_df['text'].dropna(inplace=True)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently

test_df['text'] = [entry.lower() for entry in test_df['text']]

# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words

test_df['text']= [word_tokenize(entry) for entry in test_df['text']]

# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))
# check test data set after cleaning 

test_df.head()
train_df[['target','text_cleaned']].head()
# Define X and y

X = train_df['text_cleaned']

y = train_df['target']
# test and train split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
#transform X_train and X_test to vectorized X_train_Tfidf and X_test_Tfidf.

Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(X)



X_train_vec = Tfidf_vect.transform(X_train)

X_test_vec = Tfidf_vect.transform(X_test)
# look at vectorized data

print(X_train_vec)
# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(X_train_vec,y_train)

# predict the labels on validation dataset

predictions_NB = Naive.predict(X_test_vec)

# Use accuracy_score function to get the accuracy

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, y_test)*100)
file_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
Vector_test = Tfidf_vect.transform(test_df['text'])
# Try naive model on test data set

y_predict_test = Naive.predict(Vector_test)

file_submission.target = y_predict_test

file_submission.to_csv("submission_naive.csv", index=False)
# check head for test data set

file_submission.head(10)
# Classifier - Algorithm - SVM

# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(X_train_vec,y_train)

# predict the labels on validation dataset

predictions_SVM = SVM.predict(X_test_vec)

# Use accuracy_score function to get the accuracy

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
# Try SVM model on test data set 

y_predict_test = SVM.predict(Vector_test)

file_submission.target = y_predict_test

file_submission.to_csv("submission_svm.csv", index=False)
# check head for test data set

file_submission.head(10)
# Classifier - Algorithm - Logistic Regression

# fit the training dataset on the classifier

logreg = LogisticRegression()

logreg.fit(X_train_vec, y_train)

# predict the labels on validation dataset

predictions_logreg = logreg.predict(X_test_vec)

# Use accuracy_score function to get the accuracy

print("Logistic regression Accuracy Score -> ",accuracy_score(predictions_logreg, y_test)*100)
# Try logreg model on test data set 

y_predict_test = logreg.predict(Vector_test)

file_submission.target = y_predict_test

file_submission.to_csv("submission_logreg.csv", index=False)
# check head for test data set

file_submission.head(10)
# Classifier - Algorithm - XGBClassifier

# fit the training dataset on the classifier

from xgboost import XGBClassifier

# fit model no training data

xgboost = XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.1,  

                      colsample_bytree = 0.4,

                      subsample = 1,

                      objective='binary:logistic', 

                      n_estimators=500, 

                      reg_alpha = 0.3,

                      max_depth=10, 

                      gamma=5)

xgboost.fit(X_train_vec, y_train)

# predict the labels on validation dataset

predictions_xgboost = xgboost.predict(X_test_vec)

# Use accuracy_score function to get the accuracy

print("XG Boost Accuracy Score -> ",accuracy_score(predictions_xgboost, y_test)*100)
# Try xgboost model on test data set  

y_predict_test = xgboost.predict(Vector_test)

file_submission.target = y_predict_test

file_submission.to_csv("submission_xgboost.csv", index=False)
# check head for test data set

file_submission.head(10)