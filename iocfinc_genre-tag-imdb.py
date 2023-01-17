# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import matplotlib.pyplot as plt
import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
# Read the dataset
data = pd.read_csv('../input/IMDB-Movie-Data.csv')
# Basic check of the data
data.info()
# Checking the data
data.head(10)
# Pre-processing: We want to at do a lower case of Genre and Description columns.
# Also added removal of punctuation.
translator = str.maketrans('','',string.punctuation)

data['Description']= data['Description'].str.lower().str.translate(translator)
data['Genre']= data['Genre'].str.lower()
data.head()
# This is for splitting the individual grouped genre into individual genre
new_data = pd.DataFrame(columns = ['Title','Genre','Description'])
for i in range(len(data['Genre'])):  # GO over the Genre
    for word in data['Genre'][i].split(","): # We will split the Genre
        new_data = new_data.append({'Title':data['Title'][i],'Genre':word,'Description':data['Description'][i]}, ignore_index = 1)
# Checking the new data created
new_data.info()
new_data
# This is for splitting the individual grouped genre into individual genre
new_data = pd.DataFrame(columns = ['Title','Genre','Description'])
for i in range(len(data['Genre'])):  # GO over the Genre
    for word in data['Genre'][i].split(","): # We will split the Genre
        new_data = new_data.append({'Title':data['Title'][i],'Genre':word,'Description':data['Description'][i]}, ignore_index = 1)
# Checking the new data created
new_data.info()
new_data.head(5)
Genre_count = Counter(new_data['Genre'])
Genre_count
# Aggregate all Genres with less that 100 items as 'others'
others = ['animation','family','music','history','western','war','musical','sport','biography']
for i in range(len(new_data['Genre'])):
    if new_data['Genre'][i] in others:
        new_data.iloc[i]['Genre'] = 'others'
new_data[new_data['Genre']=='others']
import re

def cleanup(string):
    '''
    Helper Function:
    Will clean up the input string (for description) in this case.
    '''
    string = re.sub(r"\n",'',string)
    string = re.sub(r"\n",'',string)
    string = re.sub(r"[0-9]",'digit',string) # We do not care for the specific number
    string = re.sub(r"\''",'',string)
    string = re.sub(r'\"','',string)
    return string.strip().lower()
X = []

for item in range(new_data.shape[0]):
    X.append(cleanup(new_data.iloc[item][2]))
y = np.array(new_data['Genre'])
X
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 5)
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
model = Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])
#Selecting the best parameter via gridsearch
from sklearn.grid_search import GridSearchCV
parameters = {'vectorizer__ngram_range':[(1,1), (1,2),(2,2)],
              'vectorizer__min_df':[0,0.001],
              'tfidf__use_idf':('True','False')}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs= -1)
gs_clf_svm = gs_clf_svm.fit(X,y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)
model = Pipeline([('vectorizer',CountVectorizer(ngram_range = (1,2),min_df=0)),
                  ('tfidf',TfidfTransformer(use_idf=True)),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])
model.fit(X,y)
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred,y_test)
model.score(X,y)
model.predict(['Robert McCall serves an unflinching justice for the exploited and oppressed, but how far will he go when that is someone he loves?'])[0]

## Description is from 'The Equalizer' Genre: Action, Crime, Thriller 
a = Counter(new_data['Genre'])
s = set(a)
s
trimmed = pd.DataFrame(columns=new_data.columns)
for genre in s:
    trimmed=trimmed.append(new_data[new_data['Genre'] == genre][:100])
trimmed.info()
trimmed[trimmed['Genre']=='action'].head()
import re

def cleanup(string):
    '''
    Helper Function:
    Will clean up the input string (for description) in this case.
    '''
    string = re.sub(r"\n",'',string)
    string = re.sub(r"\n",'',string)
    string = re.sub(r"[0-9]",'digit',string) # We do not care for the specific number
    string = re.sub(r"\''",'',string)
    string = re.sub(r'\"','',string)
    return string.strip().lower()
X = []

for item in range(trimmed.shape[0]):
    X.append(cleanup(trimmed.iloc[item][2]))
y = np.array(trimmed['Genre'])
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 5)
# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
model = Pipeline([('vectorizer',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])
#Selecting the best parameter via gridsearch
from sklearn.grid_search import GridSearchCV
parameters = {'vectorizer__ngram_range':[(1,1), (1,2),(2,2)],
              'vectorizer__min_df':[0,0.001],
              'tfidf__use_idf':('True','False')}
gs_clf_svm = GridSearchCV(model, parameters, n_jobs= -1)
gs_clf_svm = gs_clf_svm.fit(X,y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)
model = Pipeline([('vectorizer',CountVectorizer(ngram_range = (1,2),min_df=0)),
                  ('tfidf',TfidfTransformer(use_idf=True)),('clf',OneVsRestClassifier(LinearSVC(class_weight='balanced')))])
model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred,y_test)
model.score(X,y)
Description_=input()
model.predict([Description_])[0]
import pickle
filename = "model-genre-classifier.sav"
pickle.dump(model, open(filename,'wb'))
loaded_model = pickle.load(open(filename,'rb'))
loaded_model.score(X,y)
### STOP HERE
