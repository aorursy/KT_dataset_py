#Pass warning notification

def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn

warnings.filterwarnings('ignore')



#Read the .csv files as pandas dataframe

import numpy as np

import pandas as pd



train_raw = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')
#Preprocess the data to fit for the classifier

from sklearn.preprocessing import LabelEncoder, StandardScaler



le = LabelEncoder().fit(train_raw.label)

labels = le.transform(train_raw.label)

classes = list(le.classes_)

test_ids = test_raw.id



train = train_raw.drop(['id', 'label'], axis=1)

test = test_raw.drop(['id'], axis=1)
#Construct the iterator

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1337)

sss.get_n_splits(train, labels)



for train_index, test_index in sss.split(train, labels):   

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.naive_bayes import *



classifiers = [

    GaussianNB(),

    MultinomialNB(),

    ComplementNB(),

    BernoulliNB()]



#Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



print('NAIVE BAYES Classifiers\n')

for clf in classifiers:

    clf = clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    print ('ML Model: ', name)



    #Cross-validation

    scores = cross_val_score(clf, train.values, labels, cv=sss)

    print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))



    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print ('Accuracy: {:.5%}'.format(acc))



    train_predictions = clf.predict_proba(X_test)

    logloss = log_loss(y_test, train_predictions)

    print ('Log Loss: {:.5}\n'.format(logloss))

    

    log_entry = pd.DataFrame([[name, acc*100, logloss]], columns=log_cols)

    log = log.append(log_entry)
#Visual comparison

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy (higher is better)')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss (lower is better)')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier



classifiers = [

    KNeighborsClassifier(3),

    LinearDiscriminantAnalysis(),

    MLPClassifier()]



print('KNeighborsClassifier -- LinearDiscriminantAnalysis -- MLPClassifier\n')

for clf in classifiers:

    clf = clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    print ('ML Model: ', name)

    

    #Cross-validation

    scores = cross_val_score(clf, train.values, labels, cv=sss)

    print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))

    

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print ('Accuracy: {:.5%}'.format(acc))



    train_predictions = clf.predict_proba(X_test)

    logloss = log_loss(y_test, train_predictions)

    print ('Log Loss: {:.5}\n'.format(logloss))
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(multi_class="multinomial", solver='newton-cg')



clf = clf.fit(X_train, y_train)

name = clf.__class__.__name__

print ('ML Model: ', name)



#Cross-validation

scores = cross_val_score(clf, train.values, labels, cv=sss)

print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))



train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print ('Accuracy: {:.4%}'.format(acc))



train_predictions = clf.predict_proba(X_test)

logloss = log_loss(y_test, train_predictions)

print ('Log Loss: {:.6}'.format(logloss))
clf = LogisticRegression(C=1000, multi_class="multinomial", tol=0.0001, solver='newton-cg')



clf = clf.fit(X_train, y_train)

name = clf.__class__.__name__

print ('ML Model: ', name)



#Cross-validation

scores = cross_val_score(clf, train.values, labels, cv=sss)

print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))



train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print ('Accuracy: {:.4%}'.format(acc))



train_predictions = clf.predict_proba(X_test)

logloss = log_loss(y_test, train_predictions)

print ('Log Loss: {:.6}'.format(logloss))
from sklearn.model_selection import GridSearchCV



#Standardize the training data.

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)



param_grid = {'C': [2310],

              'tol': [0.0001]}



clf = LogisticRegression(solver='newton-cg', multi_class='multinomial')



grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', refit='True', n_jobs=1, cv=sss)

grid_search.fit(X_train_scaled, y_train)



print ('Best parameter: {}'.format(grid_search.best_params_))

print ('Best cross-validation neg_log_loss score: {}'.format(grid_search.best_score_))

print ('\nBest estimator:\n{}'.format(grid_search.best_estimator_))
scaler = StandardScaler().fit(X_test)

X_test_scaled = scaler.transform(X_test)



name = clf.__class__.__name__

print ('ML Model: ', name + ' + GridSearch')



#Cross-validation

scores = cross_val_score(clf, train.values, labels, cv=sss)

print ('Mean Cross-validation scores: {}'.format(np.mean(scores)))



train_predictions = grid_search.predict(X_test_scaled)

acc = accuracy_score(y_test, train_predictions)

print ('Accuracy: {:.4%}'.format(acc))



train_predictions_p = grid_search.predict_proba(X_test_scaled)

logloss = log_loss(y_test, train_predictions_p)

print ('Log Loss: {:.6}'.format(logloss))
#Predict Test Set

favorite_clf = LogisticRegression(C=2310, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='multinomial',

          n_jobs=None, penalty='l2', random_state=None, solver='newton-cg',

          tol=0.0001, verbose=0, warm_start=False)

favorite_clf.fit(X_train, y_train)

test_predictions = favorite_clf.predict_proba(test)



#Format DataFrame

submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()



#Export Submission

submission.to_csv('submission.csv', index=False)

submission.tail()
from IPython.display import HTML

import base64



def create_download_link(submission, title = "Download CSV di sini bang!", filename = "LogReg-GridSearch[C=2310,tol=0.0001]-Rand1337.csv"):  

    csv = submission.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(submission)