import pandas as pd 

import numpy as np

import re

import seaborn as sns

from bs4 import BeautifulSoup



from sklearn import model_selection, ensemble, metrics

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



import nltk

nltk.download('stopwords', 'wordnet', 'punkt')

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)

test_data = pd.read_csv("../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)

submit_data = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3)
print('Shape of train data: ', train_data.shape)

print('Shape of test data: ', test_data.shape)

print('Shape of submit data: ', submit_data.shape)
train_data.head()
print('Column names:', list(train_data.columns))
train_data.review[0]
def my_tokenizer(sample):

    # Split into words

    words = nltk.word_tokenize(sample)

#     print('3',words)

    

    # Leave alphabetical tokens

    tokens = [word for word in words if word.isalnum()]

    tokens = [word for word in tokens if not word.isdigit()]

#     print('4',tokens)

    

    # Remove stopwords

    meaningful_words = [w for w in tokens if not w in stops]

#     print('5',meaningful_words)

    

    # Lemmatization 

    word_list = [lemmatizer.lemmatize(w) for w in meaningful_words]

#     print('6',word_list)

    

    return word_list



def my_preprocessor(sample):

    # Remove HTML tags

    no_tags_text = BeautifulSoup(sample).get_text()  

#     print('1',no_tags_text)

    

    # To lowercase

    review_text = no_tags_text.lower()

#     print('2',review_text)

    

    return review_text
stops = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

vectorizer = CountVectorizer(analyzer = "word", tokenizer = my_tokenizer, preprocessor = my_preprocessor, \

                             stop_words = None, max_features = 5000) 
sample = train_data.review[0]

train_data_features = vectorizer.fit_transform([sample])

train_data_features = train_data_features.toarray()



train_data_features
train_data_features = vectorizer.fit_transform(train_data.review)

train_data_features = train_data_features.toarray()
# test_data_features = vectorizer.transform(test_data.review)

# test_data_features = test_data_features.toarray()
submit_data_features = vectorizer.transform(submit_data.review)

submit_data_features = submit_data_features.toarray()
train_data_features.shape
vocab = vectorizer.get_feature_names()
roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score)

X_train = train_data_features[:500]

Y_train = train_data.sentiment[:500]
forest = RandomForestClassifier(n_estimators = 100) 

forest = forest.fit( train_data_features, train_data.sentiment )
result = forest.predict(submit_data_features)
params = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 5, 10]}

svc = SVC(probability = True, random_state = 0)

clf = GridSearchCV(svc, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

clf.fit(X_train, Y_train)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
svc_best = SVC(C = clf.best_params_['C'], kernel = clf.best_params_['kernel'], probability = True, random_state = 0)
params = {'n_estimators':[10, 50, 100, 150], 'criterion':['gini', 'entropy'], 'max_depth':[None, 5, 10, 50]}

rf = RandomForestClassifier(random_state = 0)

clf = GridSearchCV(rf, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

clf.fit(X_train, Y_train)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
rf_best = RandomForestClassifier(n_estimators = clf.best_params_['n_estimators'], criterion = clf.best_params_['criterion'], \

                                 max_depth = clf.best_params_['max_depth'], random_state = 0)
params = {'penalty':['l1', 'l2'], 'C':[1, 2, 3, 5, 10]}

lr = LogisticRegression(random_state = 0)

clf = GridSearchCV(lr, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

clf.fit(X_train, Y_train)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
lr_best = LogisticRegression(penalty = clf.best_params_['penalty'], C = clf.best_params_['C'], random_state = 0)

# lr_best = LogisticRegression(penalty = 'l2', C = 10, random_state = 0)
params = {"var_smoothing" : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}

nb = GaussianNB()

clf = GridSearchCV(nb, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

clf.fit(X_train, Y_train)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
nb_best = GaussianNB(var_smoothing = clf.best_params_['var_smoothing'])
params = {'n_neighbors':[3, 5, 10, 20], 'p':[1, 2, 5], 'weights':['uniform', 'distance']}

knc = KNeighborsClassifier()

clf = GridSearchCV(knc, param_grid = params, scoring = roc_auc_scorer, cv = 5, n_jobs = -1)

clf.fit(X_train, Y_train)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
knc_best = KNeighborsClassifier(n_neighbors = clf.best_params_['n_neighbors'], p=clf.best_params_['p'],\

                               weights = clf.best_params_['weights'])
# voting_clf = VotingClassifier(estimators=[('svc', svc_best), ('rf', rf_best), ('lr', lr_best), ('nb', nb_best),\

#                                           ('knc', knc_best)], voting='hard')

# voting_clf.fit(train_data_features, train_data.sentiment)

lr_best.fit(train_data_features, train_data.sentiment)

y_pred = lr_best.predict(submit_data_features)
submission = pd.read_csv("../input/word2vec-nlp-tutorial/sampleSubmission.csv", header=0, delimiter=",", quoting=3)

col = submission.columns[1]

submission[col] = y_pred

submission.to_csv('submission.csv', index=False)
f = open("submission.csv", "r")

f.readline()

s = open("valid_submission.csv","w+")

s.write('\"id\",\"sentiment\"\n')

for x in f:

    x = x.split(',')

    x[0] = x[0][2:-2]

    s.write(','.join(x))