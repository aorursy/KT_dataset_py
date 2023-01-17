# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords, brown
from nltk import word_tokenize
from nltk.util import ngrams
import math
stop_words = set(stopwords.words('english'))
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
#importing the data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
#nlp/machine learning libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,GridSearchCV
from sklearn.metrics import f1_score,classification_report,roc_curve,precision_recall_curve,auc,average_precision_score
from sklearn.feature_selection import chi2, SelectKBest
import re
import pandas, xgboost, numpy, textblob, string
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
#display top 5 rows
train_df.head()
test_df.head()
#features
X = train_df['question_text']
#target label
Y = train_df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
#pipeline for creating tf idf and  basic logistic regression model
baseline_ngram_lr = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words,ngram_range=(1,3))),
                    ('classifier', LogisticRegression()),
                    ])
#fitting the pipeline to the train data
baseline_ngram_lr.fit(X_train, y_train )
baseline_ngram_lr_preds = baseline_ngram_lr.predict(X_test)
print(classification_report(y_test, baseline_ngram_lr_preds))
baseline_ngram_lr_preds_prob = baseline_ngram_lr.predict_proba(X_test)[:,1]
f1_list = []
for threshold in np.arange(0.1, 0.6, 0.01):
    threshold = np.round(threshold, 2)
    f1_list.append((f1_score(y_test, (baseline_ngram_lr_preds_prob>threshold).astype(int)),threshold))
    print("F1 score at threshold {0} is {1}".format(threshold, f1_score(y_test, (baseline_ngram_lr_preds_prob>threshold).astype(int))))
def sort_tuple(tup):
    return tup[0]

best_threshold = sorted(f1_list,key=sort_tuple, reverse=True)[0][1]
##creating a submission file with the optimal threshold with the baseline model
def submission(df, predictions, file_name, threshold=0.20):
    print('Optimal threshold with better F1 score is: ', threshold)
    results = (predictions > threshold).astype(int)
    df['prediction'] = results
    file = (file_name + '.csv')
    df.to_csv(file, index=False)
#predicting the classes on test data
baseline_ngram_lr_preds_prob = baseline_ngram_lr.predict_proba(test_df['question_text'])
print('Saving the results in the submission file')
sub_df = pd.read_csv('../input/sample_submission.csv')
submission(sub_df, baseline_ngram_lr_preds_prob, 'submission', threshold=best_threshold)
print("At threshold {0}, we are getting better F1 score and we will be choosing this threshold for our submission. This is our baseline and we will try to beat this score".format(best_threshold))
explainer = LimeTextExplainer(class_names=['sincere','insincere'])
idx = 117
exp = explainer.explain_instance(test_df['question_text'][idx], baseline_ngram_lr.predict_proba, num_features=5)
exp.show_in_notebook(text=test_df['question_text'][idx])
idx =  56368
exp = explainer.explain_instance(test_df['question_text'][idx], baseline_ngram_lr.predict_proba, num_features=5)
exp.show_in_notebook(text=test_df['question_text'][idx])
