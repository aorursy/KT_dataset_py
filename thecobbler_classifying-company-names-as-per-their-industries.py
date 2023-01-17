# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# libraries for dataset preparation, feature engineering, model training 

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble

from sklearn.linear_model import LogisticRegression

import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, Model

from keras.layers import Activation, Dense, Dropout

from sklearn.preprocessing import LabelBinarizer

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multiclass import OneVsRestClassifier

import time

import os



#Visualization

import matplotlib.pyplot as plt 

import seaborn as sns



#Preprocessing related libraries

import string

import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_excel(r"../input/73stringscasestudyround/Training Data.xlsx")
train.head(2)
#Identifying the first Tag for all rows

first_tag = []

for str in train['Industry Classifications']:

    first_tag.append(str.split(';')[0])
#Number of Records in First_Tag

len(first_tag)
#Adding a new feature into the training dataset

train['First_Tag'] = first_tag
train.head(2)
#Unique Values for Geographic Locations

train['Geographic Locations'].unique()
#Unique Values for CompanyStatus

train['Company Status'].unique()
#Unique Values for CompanyType

train['Company Type'].unique()
#Abstracting CompanyName,BusinessDescription,IndustryClassification from train into a new dataset

train_ana = train[['Company Name','Business Description','First_Tag']]
train_ana.head(2)
#Adding and additional field named " Tidy Description", where we will store the transformed description

train_ana['Tidy_Desc'] = train_ana['Business Description'].str.lower()
train_ana.head(1)
import re

#Function to remove any additional special characters, if needed.

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt
#Facts: Translate function has been changed from Python 2.x to Python 3.x

#It now takes only one output

train_ana.Tidy_Desc = train_ana.Tidy_Desc.apply(lambda x: x.translate({ord(c):'' for c in "1234567890"}))
train_ana.head(1)
#[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]

train_ana.Tidy_Desc=train_ana.Tidy_Desc.apply(lambda x: x.translate({ord(c):'' for c in "[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]"}))
train_ana.head(1)
train_ana.Tidy_Desc=train_ana.Tidy_Desc.apply(lambda x: x.strip())
train_ana.head(1)
stop_words = stopwords.words('english')

from nltk.tokenize import word_tokenize
# function to remove stopwords

def remove_stopwords(sen):

    sen_new = " ".join([i for i in sen if i not in stop_words])

    return sen_new
clean_sentences  = [remove_stopwords(r.split()) for r in train_ana['Tidy_Desc']]
train_ana['Tidy_Desc'] = clean_sentences
train_ana.head(1)
#Tokenization

tokens = train_ana['Tidy_Desc'].apply(lambda x: x.split())

#Now that we have the removed StopWords and tokenized the Business Descriptions

#We will now subject the tokenized version to removing stop words, sparse terms, and particular words

#In some cases, it’s necessary to remove sparse terms or particular words from texts. 

#This task can be done using stop words removal techniques considering that any group of words can be chosen as the stop words.
#Stemming

from nltk.stem import PorterStemmer

stemmer= PorterStemmer()

Stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
for i in range(len(Stemmed_tokens)):

    Stemmed_tokens[i] = ' '.join(Stemmed_tokens[i])



train_ana['Tidy_Desc_Stemmed'] = Stemmed_tokens
train_ana.head(1)
#Lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

Lemmatized_tokens = tokens.apply(lambda x: [lemmatizer.lemmatize(i) for i in x]) # Lemmatizing
for i in range(len(Lemmatized_tokens)):

    Lemmatized_tokens[i] = ' '.join(Lemmatized_tokens[i])



train_ana['Tidy_Desc_Lemma'] = Lemmatized_tokens
train_ana.head(1)
#Before going for Count Vectorization as Feature

#We would like to check the type of Count of Type of Tags



train_ana['First_Tag'].value_counts().head(5)
#Removing the rows with missing Description

train_ana = train_ana[train_ana['First_Tag'] != '-']
#set(train_ana['First_Tag'])
#Selecting the first 10 labels

T10Tag = train_ana['First_Tag'].value_counts().index.tolist()
T10Tag = T10Tag[:10]
T10Tag
train_10 = train_ana[train_ana['First_Tag'].isin(T10Tag)]
train_10.shape
set(train_10['First_Tag'])
from collections import Counter

Counter(train_10['First_Tag'])
train_10.columns.tolist()
train_10_ana = train_10[['Tidy_Desc_Lemma','First_Tag']]
#pre-processing

import re 

def clean_str(string):

    """

    Tokenization/string cleaning for dataset

    Every dataset is lower cased except

    """

    string = re.sub(r"\n", "", string)    

    string = re.sub(r"\r", "", string) 

    string = re.sub(r"[0-9]", "digit", string)

    string = re.sub(r"\'", "", string)    

    string = re.sub(r"\"", "", string)    

    return string.strip().lower()
#train test split

from sklearn.model_selection import train_test_split

X = []

for i in range(train_10_ana.shape[0]):

    X.append(clean_str(train_10_ana.iloc[i][0]))

y = np.array(train_10_ana["First_Tag"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
#feature engineering and model selection

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
#pipeline of feature engineering and model

model = Pipeline([('vectorizer', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
#paramater selection

from sklearn.model_selection import GridSearchCV

parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],

               'tfidf__use_idf': (True, False)}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)

gs_clf_svm = gs_clf_svm.fit(X, y)

print(gs_clf_svm.best_score_)

print(gs_clf_svm.best_params_)

#preparing the final pipeline using the selected parameters

model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),

    ('tfidf', TfidfTransformer(use_idf=True)),

    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])
#fit model with training data

model.fit(X_train, y_train)
#evaluation on test data

pred = model.predict(X_test)
model.classes_
from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(pred, y_test)
accuracy_score(y_test, pred)