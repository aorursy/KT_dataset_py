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





import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings

warnings.filterwarnings("ignore")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

df.head()
df['Category'].isnull().any()
df = df.loc[df['Message'].notna(),:]
text = df['Message']
labels = df['Category']

# tokenize

text = text.apply(nltk.word_tokenize)

print('tokenized')



# Remove stop words

stop_words = set(nltk.corpus.stopwords.words('english'))

text = text.apply(lambda x : [item for item in x if item not in stop_words])

print('stop words removed')



#Remove punctuation

regex = '[a-z]+'

text = text.apply(lambda x: [item for item in x if re.match(regex, item)])

print('puncuation,numbers,symbols removed')



# Lemmatization

lem = nltk.stem.wordnet.WordNetLemmatizer()

text = text.apply(lambda x : [lem.lemmatize(item,pos='v') for item in x])

print('lemmatized')
# Classification using TFIDF vectorizer

text=[" ".join(sen) for sen in text.values]

# Vectorize training and testing data

def Vectorize(vec, X_train, X_test):    

    

    X_train_vec = vec.fit_transform(X_train)

    X_test_vec = vec.transform(X_test)

    

    print('Vectorization complete.\n')

    

    return X_train_vec, X_test_vec



# Use multiple classifiers and grid search for prediction

def ML_modeling(models, params, X_train, X_test, y_train, y_test):    

    

    if not set(models.keys()).issubset(set(params.keys())):

        raise ValueError('Some estimators are missing parameters')



    for key in models.keys():

    

        model = models[key]

        param = params[key]

        gs = GridSearchCV(model, param, cv=5, error_score=0, refit=True)

        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)

        

        # Print scores for the classifier

        print(key, ':', gs.best_params_)

        print("Precision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')))

    

    return



models = {

    'Naive Bayes': MultinomialNB(), 

    'Decision Tree': DecisionTreeClassifier(),  

    'Perceptron': MLPClassifier(),

    'Gradient Boosting': GradientBoostingClassifier()

}



params = {

    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] }, 

    'Decision Tree': { 'min_samples_split': [1, 2, 5] }, 

    'Perceptron': { 'alpha': [0.0001, 0.001], 'activation': ['tanh', 'relu'] },

    'Gradient Boosting': { 'learning_rate': [0.05, 0.1], 'min_samples_split': [2, 5] }

}







# Train-test split and vectorize

X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, shuffle=True)

X_train_vec, X_test_vec = Vectorize(TfidfVectorizer(), X_train, X_test)



ML_modeling(models, params, X_train_vec, X_test_vec, y_train, y_test)