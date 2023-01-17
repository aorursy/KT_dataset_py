%reload_ext autoreload

%autoreload 2

%matplotlib inline



import os

import re

import string

import pandas as pd

import numpy as np

import sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import stop_words

from sklearn.metrics import log_loss, make_scorer

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



from sklearn.grid_search import GridSearchCV

from sklearn.grid_search import RandomizedSearchCV

train_file = "../input/train.csv"

test_file = "../input/test.csv"
test = pd.read_csv(test_file)
train = pd.read_csv(train_file)
train.head(5)
train['is_duplicate'].value_counts()
train[train['is_duplicate']==1].head(5)
train.dropna(inplace = True)
train.shape
def tokenize(text):

    """

    Given a string, return a list of words normalized as follows.

    Split the string to make words first by using regex compile() function

    and string.punctuation + '0-9\\r\\t\\n]' to replace all those

    char with a space character.

    Split on space to get word list.

    Ignore words < 3 char long.

    Lowercase all words

    Remove English stop words

    """

    re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    regex = re.compile(re.escape(string.punctuation) + '0-9\\r\\t\\n]')

    

    

    regex = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])'+re.escape(string.punctuation) + '0-9\\r\\t\\n]')

    words = regex.sub(r' \1 ', text).split()

    words = [w.lower() for w in words]

    words = [w.strip() for w in words]

    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...

    words = [w for w in words if w not in stop_words.ENGLISH_STOP_WORDS]

    return words

veczr = CountVectorizer(lowercase = True, analyzer= 'word', stop_words='english', min_df = 0.0001, max_features= 3000)
voc = veczr.fit(pd.concat([train['question1'],train['question2']])) # Learn a vocabulary dictionary of all tokens in both questions 
q1 = voc.transform(train['question1']) # Transform documents (question 1) to document-term matrix 

q2 = voc.transform(train['question2']) # Same for question 2
from scipy.sparse import hstack



X = hstack((q1,q2)) # stacking together two matrices

X.shape

# Convert from coo to csr format

X = X.toarray()
y = train['is_duplicate']

y.shape
x_train,x_val,y_train, y_val= train_test_split(X,train['is_duplicate'], test_size = 0.2, random_state =42)
print(x_train.shape)

print(y_train.shape)
vocab = veczr.get_feature_names(); vocab[800:900]
train.loc[390,'question1'].split()
veczr.vocabulary_['wedding'] # find the position of the word 'wedding'
X[390,2931] # Yep!
mlr = LogisticRegression(C = 0.1, dual = True, n_jobs = -1)

mlr.fit(x_train,y_train)
# make predictions

pred_train = mlr.predict(x_train)

pred_prob_train = mlr.predict_proba(x_train)

pred_val = mlr.predict(x_val)

pred_prob_val = mlr.predict_proba(x_val)

print("Accuracy of training:",(pred_train.T == y_train).mean())

print("log-loss of training is:", log_loss(y_train,pred_prob_train))

print("Accuracy of validation:",(pred_val.T == y_val).mean())

print("log-loss of validation is:", log_loss(y_val,pred_prob_val))
# use binarized version

mlr = LogisticRegression(C = 0.1, dual = True, n_jobs = -1)

mlr.fit(x_train.sign(),y_train)

pred_train = mlr.predict(x_train.sign())

pred_prob_train = mlr.predict_proba(x_train.sign())

pred_val = mlr.predict(x_val.sign())

pred_prob_val = mlr.predict_proba(x_val.sign())

print("Accuracy of training:",(pred_train.T == y_train).mean())

print("log-loss of training is:", log_loss(y_train,pred_prob_train))

print("Accuracy of validation:",(pred_val.T == y_val).mean())

print("log-loss of validation is:", log_loss(y_val,pred_prob_val))
veczr1 = CountVectorizer(lowercase = True, ngram_range= (1,3), stop_words='english', min_df = 0.0001, max_features= 3000)

voc1 = veczr1.fit(pd.concat([train['question1'],train['question2']])) # Learn a vocabulary dictionary of all tokens in both questions 
q1 = voc1.transform(train['question1']) # Transform documents (question 1) to document-term matrix 

q2 = voc1.transform(train['question2']) # Same for question 2
X1 = hstack([q1,q2]) # statcking together two matrices

X1.shape
x1_train,x1_val,y1_train, y1_val= train_test_split(X1,train['is_duplicate'], test_size = 0.2, random_state =42)
# use binarized version

mlr1 = LogisticRegression(C = 0.1, dual = True, n_jobs = -1)

mlr1.fit(x1_train.sign(),y1_train)

pred_train = mlr.predict(x1_train.sign())

pred_prob_train = mlr.predict_proba(x1_train.sign())

pred_val = mlr.predict(x1_val.sign())

pred_prob_val = mlr.predict_proba(x1_val.sign())

print("Accuracy of training:",(pred_train.T == y1_train).mean())

print("log-loss of training is:", log_loss(y1_train,pred_prob_train))

print("Accuracy of validation:",(pred_val.T == y1_val).mean())

print("log-loss of validation is:", log_loss(y1_val,pred_prob_val))
mrf = RandomForestClassifier(n_estimators = 50, min_samples_leaf = 3, max_features = 0.5, n_jobs = -1, max_depth = 20, random_state = 42, oob_score = True)

mrf.fit(x_train,y_train)
def print_score(m):

    res = [log_loss(y_train,m.predict_proba(x_train)), log_loss(y_val,m.predict_proba(x_val)),

                m.score(x_train, y_train), m.score(x_val, y_val)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
param_grid = {#"n_estimators": np.arange(25, 100, 25,dtype=int)}

              "max_depth": np.arange(45, 105, 10)} 

              #"min_samples_split": np.arange(1,150,1),

              #"min_samples_leaf": [10,50,100]}

              #"max_leaf_nodes": np.arange(2,60,6),

              #"min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
m = RandomForestClassifier(random_state=42, n_estimators = 100, n_jobs = -1, oob_score=True)

m.fit(x_train, y_train)
random_cv = RandomizedSearchCV(m, param_distributions = param_grid, cv = 3, scoring=log_loss_scorer)

random_cv.fit(x_train, y_train)



print(random_cv.best_score_)

print(random_cv.best_params_)

print(random_cv.best_estimator_)    
from sklearn.model_selection import GridSearchCV



grid_search = GridSearchCV(m, param_grid=param_grid, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

print(grid_search.cv_results_)

result = grid_search.cv_results_
grid_search.n_splits_
md_score = result['mean_test_score']

md = range(1,50,5)
import matplotlib.pyplot as plt

plt.plot(md,md_score)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

plt.show()