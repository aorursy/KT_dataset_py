!pip install scikit-optimize --upgrade
import os

import random

import re

import time



import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier 



import skopt

from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.utils import use_named_args



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file_path = '/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv'

data = pd.read_csv(file_path)
data.head()
data['Body'] = data['Title'] + " " + data['Body']
data.head()
# Clean the data

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^(a-zA-Z)\s]','', text)

    return text



data['Body'] = data['Body'].apply(clean_text)

data.head()
#def remove_stopword

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

  

example_sent = "This is a sample sentence, showing off the stop words filtration."  

stop_words = set(stopwords.words('english')) 



def remove_stopword(words):

    list_clean = [w for w in words.split(' ') if not w in stop_words]

    return ' '.join(list_clean)



data['Body'] = data['Body'].apply(remove_stopword)

data.head()
data.shape
N = len(data)

TRAIN_PERC = 0.8

ind_train = np.random.rand(N) < TRAIN_PERC

train, test = data[ind_train], data[~ind_train]

print(f'len(train)={len(train)}; len(test)={len(test)}')
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(train.Body)

X_train_counts.shape
X_test_counts = count_vect.transform(test.Body)

X_test_counts.shape
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

X_test_tfidf.shape
# I tried to add the length of Title, length of Body and number of tags, but it seems to globally decrease the accuracy!

"""

from scipy.sparse import hstack



def add_len_feat(X_tfidf, train_or_test):

    list_title_len = [[len(title)] for title in train_or_test.Title]

    list_body_len = [[len(body)] for body in train_or_test.Body]

    list_tag_len = [[len(tag)] for tag in train_or_test.tags_processed]

    return hstack([X_tfidf, list_title_len, list_body_len, list_tag_len])



X_train_tfidf = add_len_feat(X_train_tfidf, train)

X_test_tfidf = add_len_feat(X_test_tfidf, test)

"""
clf_dict = {

    'LogisticRegression': LogisticRegression,

    'MultinomialNB': MultinomialNB,

    'DecisionTreeClassifier': DecisionTreeClassifier,

    'SGDClassifier': SGDClassifier,

    'Perceptron': Perceptron,

    'RidgeClassifier': RidgeClassifier,

    'LinearSVC': LinearSVC,

    'RandomForestClassifier': RandomForestClassifier,

    'GradientBoostingClassifier': GradientBoostingClassifier,

    #'MLPClassifier': MLPClassifier,

}
def get_accuracy(clf, n_estimators=None, max_depth=None, learning_rate=None, max_iter=None):

    start = time.time()

    text_clf = clf(**params).fit(X_train_tfidf, train.Y)

    predicted = text_clf.predict(X_test_tfidf)

    print(f'Accuracy gets in {round(time.time()-start, 2)}s.')

    return np.mean(predicted == test.Y)



result_dict = {}



for clf_str, clf_fn in clf_dict.items():

    if clf_str == 'LogisticRegression':

        params = {'max_iter': 200}

    elif clf_str == 'RandomForestClassifier':

        params = {'n_estimators': 50,

                  'max_depth': 10}

    elif clf_str == 'DecisionTreeClassifier':

        params = {'max_depth': 10}

    elif clf_str == 'GradientBoostingClassifier':

        params = {'n_estimators': 50,

                  'learning_rate': 0.1}

    else:

        params = {}

    accuracy = get_accuracy(clf=clf_fn, **params)

    result_dict[clf_str] = accuracy

    print(f"Clf={clf_str}; Accuracy={accuracy}")
result_dict = {

    k: v

    for k, v in sorted(

        result_dict.items(),

        key=lambda x: x[1],

        reverse=True

    )

}



result_dict
DIM_Logistic = [

    Integer(100, 400, name='max_iter')

]



DIM_SVC = [

    Real(1e-5, 1, name='tol', prior='log-uniform'),

    Real(0.1, 1.5, name='C', prior='log-uniform')

]



DIM_SGDC = [

    Real(1e-5, 1e-2, name='alpha', prior='log-uniform')

]



DIM_RF = [

    Integer(1, 100, name='n_estimators'),

    Integer(5, 30, name='max_depth')

]



DIMS = {

    'LogisticRegression': DIM_Logistic,

    'LinearSVC': DIM_SVC,

    'SGDClassifier': DIM_SGDC,

    'RandomForestClassifier': DIM_RF

}
def optimize(clf_str='LinearSVC'):

    

    dimensions = DIMS[clf_str]

    print(dimensions)

    

    @use_named_args(dimensions=dimensions)

    def fitness(**params):

        clf = clf_dict[clf_str](**params)

        text_clf = clf.fit(X_train_tfidf, train.Y)

        predicted = text_clf.predict(X_test_tfidf)

        accuracy = np.mean(predicted == test.Y)

        print(f'accuracy={accuracy} with params={params}')

        return -1.0 * accuracy

    

    res = gp_minimize(func=fitness,

                      dimensions=dimensions,

                      acq_func='EI', # Expected Improvement.

                      n_calls=10,

                      random_state=666)

    print(f'best accuracy={-1.0 * res.fun} with {res.x}')

    return res
res_dict = {}

for clf_str, clf_dim in DIMS.items():

    print(f'start optimizaton for {clf_str}')

    res = optimize(clf_str=clf_str)

    res_dict[clf_str] = res
for clf_str, res in res_dict.items():

    hyperparameters_label = [hp.name for hp in DIMS[clf_str]]

    best_hyperparameters = dict(zip(hyperparameters_label, res.x))

    print(f'clf={clf_str}\nbest accuracy={-res.fun}\nbest hyperparameters={best_hyperparameters}\n')