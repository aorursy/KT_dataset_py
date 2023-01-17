from __future__ import print_function

import logging
import numpy as np
import pandas as pd

from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.utils.extmath import density
from sklearn import metrics
import re

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=6)
#DATA_DIR='../data/cooking'
DATA_DIR='../input'
TRAIN_FILE=DATA_DIR + '/train.json'
TEST_FILE=DATA_DIR + '/test.json'
SUBMIT_FILE= 'sample_submission.csv'

def clean_data(X):
    X['ingredients'] = X['ingredients'].map(lambda l: [x.lower() for x in l])
    X['ingredients'] = X['ingredients'].map(lambda l: [re.sub(r'\(\s*[^\s]*\s+oz\.\s*\)\s*', '', x).strip() for x in l])
    X['ingredients'] = X['ingredients'].map(lambda l: [x.replace("-", " ") for x in l])
    X['ingredients'] = X['ingredients'].map(lambda l: [x.replace("half & half", "half milk cream") for x in l])
    X['ingredients'] = X['ingredients'].map(lambda l: [x.replace(" & ", " ") for x in l])
    X['ingredients'] = X['ingredients'].map(lambda l: [re.sub(r'\d+%\s+less\s+sodium ', '', x).strip() for x in l])
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    X['ingredients'] = X['ingredients'].map(lambda l: [lemmatizer.lemmatize(x) for x in l])
    
    return X
def words_to_text(list_of_words, add_separate_words=True):
    # words_to_text(df['ingredients'][10], add_separate_words=True)
    if isinstance(list_of_words, list):
        l = list_of_words
    else:
        l = eval(list_of_words)
    # concatenates list of words into a text 
    s = ''
    for i, w_0 in enumerate(l):
        if i > 0:
            s = s + ' '
        w_0 = w_0.strip()
        w_1 = w_0.replace(' ', '_').replace(',','_vir_') # avoid space cut between words
        s = s + w_1
        if add_separate_words and ' ' in w_0:
            s = s + ' ' + w_0
    return s

def vectorize(X_words, sublinear_tf=True, max_df=0.75):
    print("Extracting vectorized feats from the training data using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, 
                                 max_df=max_df,
                                 stop_words='english')
    X_vectorized = vectorizer.fit_transform(X_words)
    duration = time() - t0
    print("done in %fs " % (duration))
    print("n_samples: %d, n_features: %d" % X_vectorized.shape)
    print()
    return X_vectorized, vectorizer, duration

# gets feature_names:
# vectorizer.get_feature_names()
# #############################################################################
# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training {}: ".format(str(type(clf))))
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy score:   %0.3f" % score)

    """
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("top 10 keywords per class:")
        target_names=y_encoded_classes_names
        for i, label in enumerate(target_names):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print("%s: %s" % (label, " ".join(X_column_names[top10])))
        print()
    """

    cm = metrics.confusion_matrix(y_test, y_pred)
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print()
    clf_descr = str(clf).split('(')[0]
    ret = {
        'classifier': clf,
        'accuracy_score': score, 
        'confusion_matrix': cm, 
        'train_time': train_time, 
        'test_time': test_time, 
        'y_pred': y_pred
    }
    return ret
import itertools
np.set_printoptions(precision=2)
def plot_confusion_matrix(cm, classes,
                          normalize_axis=None, # 0 : sum of each row = 1, 1: sum of columns = 1, None
                          cmap=plt.cm.Reds # https://matplotlib.org/users/colormaps.html
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize_axis is not None:
        title = "Normalized confusion matrix (axis={})".format(normalize_axis)
        cm = cm.astype('float') / cm.sum(axis=normalize_axis)[:, np.newaxis]
    else:
        title = 'Confusion matrix, without normalization'
    print(title)
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=22)
    plt.title(title, fontsize=30)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize_axis is not None else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=16)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    plt.tight_layout()
    return cm # cm normalized

def top_feature_importances(clf, max_feats):
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:max_feats]
        names = [X_column_names[i] for i in indices]
        stds = [np.std([tree.feature_importances_[i] for tree in clf.estimators_])
               for i in indices]
        importances = importances[indices]
        return importances, names, indices, stds
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("\ntop 10 most important features per class:")
        ret = {}
        for i, label in enumerate(y_encoded_classes_names):
            importances = clf.coef_[i][-max_feats:]
            indices = np.argsort(clf.coef_[i])[-max_feats:]
            names = [X_column_names[i] for i in indices]
            print("%s :  %s\n" % (label, " ".join(names)))
            ret[label] = (importances, names, indices)
        return ret
    return None

# plot feature importances   
def plot_top_feature_importances(clf, max_feats=20, class_name=None):
    # feature importances
    top = top_feature_importances(clf, max_feats)
    if isinstance(top, list):
        # case of tree methods
        importances, feat_names, indices, stds = top
    elif isinstance(top, dict):
        # svm methods
        importances, feat_names, indices = top[class_name]
        print('importances lenth: ', len(importances))
        stds=None
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(20, int(0.6*max_feats)))
    plt.tick_params(labelsize=13)
    plt.title("{} top feature importances".format(max_feats), fontsize=18)
    #plt.bar(range(max_feats), importances, color="r", yerr=stds, align="center")
    #plt.xticks(range(max_feats), feat_names)
    plt.barh(range(max_feats), importances, color="r", yerr=stds, align="center")
    plt.yticks(range(max_feats), feat_names)
    ax = plt.gca(); ax.invert_yaxis()  # inverse y axis

df = pd.read_json(TRAIN_FILE, encoding='iso-8859-1')
# clean data
clean_data(df)
df.head(3)
X_words = df['ingredients'].map(words_to_text)
X_words.head(3)
X_vects, Vectorizer_inst, Vectorizer_duration = vectorize(X_words)
X_column_names = Vectorizer_inst.get_feature_names()
# encode target
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['cuisine'])
# classe names corresponding to encoded values
# y_encoded_classes_names = [encoder.inverse_transform(c) for c in range(20)]
y_encoded_classes_names = encoder.inverse_transform(range(20))
# using scipy.sparse.csr.csr_matrix to speed up training
from sklearn.model_selection import train_test_split
Xs_train, Xs_test, y_train, y_test = train_test_split(X_vects, y_encoded, random_state=0)
BEST_PARAMS = {
    'C': 80,
    'kernel': 'rbf',
    'gamma': 1.7,
    'coef0': 1,
    'cache_size': 500
}
BEST_ESTIMATOR = OneVsRestClassifier(
        SVC(**BEST_PARAMS))
RESULT = benchmark(BEST_ESTIMATOR, Xs_train, y_train, Xs_test, y_test)
cm = RESULT['confusion_matrix']
plt.figure(figsize=(18,18))
cm_norm = plot_confusion_matrix(cm, 
                      classes=y_encoded_classes_names,
                      normalize_axis=None, # unnormalized
                      cmap=plt.cm.RdPu)

# Show what's wrong
pd.set_option('display.max_colwidth', 120)

y_pred = RESULT['y_pred']
X_test_words = Vectorizer_inst.inverse_transform(Xs_test)
y_test_class = encoder.inverse_transform(y_test)
y_test_pred_class = encoder.inverse_transform(y_pred)
df_check = pd.DataFrame({
    'TRUE_CLASS': y_test_class,
    'PRED_CLASS': y_test_pred_class,
    'ingredients': X_test_words,
     })
df_ko = df_check[df_check['TRUE_CLASS'] != df_check['PRED_CLASS']]
print('Some misclassfied cases:')
df_ko.sample(10)
df_ko[(df_ko['PRED_CLASS'] == 'italian') & (df_ko['TRUE_CLASS'] == 'french')].head(10)
df_ko[(df_ko['PRED_CLASS'] == 'french') & (df_ko['TRUE_CLASS'] == 'italian')].head(10)
df_ko[(df_ko['PRED_CLASS'] == 'southern_us') & (df_ko['TRUE_CLASS'] == 'indian')]
# read dataset for submission
df_test_subm = pd.read_json(TEST_FILE, encoding='iso-8859-1')
df_test_subm.set_index('id', inplace=True)
df_test_subm.head(20)
# clean data
clean_data(df_test_subm)

X_test_subm_words = df_test_subm['ingredients'].map(words_to_text)
X_test_subm_vects = Vectorizer_inst.transform(X_test_subm_words)
y_test_subm_encoded = BEST_ESTIMATOR.predict(X_test_subm_vects)
# Decode class names
y_test_subm = [y_encoded_classes_names[i] for i in y_test_subm_encoded]
y_test_subm[:3]
df_test_subm['cuisine'] = y_test_subm
df_test_subm.head(20)
df_test_subm['cuisine'].to_csv(SUBMIT_FILE, header=True)